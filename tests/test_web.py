"""Offline web upload, streaming, cleanup, and validation checks."""
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app import app


def materials():
    return {
        "question_paper": (io.BytesIO(b"Question 1"), "paper.txt"),
        "model_answers_text": "1. Reference answer",
        "answer_files": [
            (io.BytesIO(b"First page"), "../../student.txt"),
            (io.BytesIO(b"Second page"), "student.txt"),
        ],
        "provider": "ollama",
        "use_critic": "on",
        "max_grade_retries": "2",
    }


def test_page_and_static_assets():
    with app.test_client() as client:
        page = client.get("/")
        assert page.status_code == 200
        assert b"Good feedback." in page.data
        assert b"streamlit" not in page.data.lower()
        assert "script-src 'self'" in page.headers["Content-Security-Policy"]
        for path in ("/static/app.js", "/static/style.css"):
            assert client.get(path).status_code == 200


def test_reviews_page_serves_review_sheet():
    with app.test_client() as client:
        page = client.get("/reviews")
        assert page.status_code == 200
        assert b"Review the evidence." in page.data
        assert b"Show reviews" in page.data
        assert "script-src 'self'" in page.headers["Content-Security-Policy"]
        assert client.get("/static/reviews.js").status_code == 200
        for control in (b"review-search", b"review-filter", b"review-sort", b"review-stats",
                        b"review-priorities", b"review-warnings", b"download-csv", b"print-reviews"):
            assert control in page.data


def test_how_it_works_page_and_diagram():
    with app.test_client() as client:
        page = client.get("/how-it-works")
        assert page.status_code == 200
        assert b"From answer sheets to" in page.data
        assert b"how-it-works" in page.data
        assert client.get("/static/img/langgraph-architecture.png").status_code == 200
        for path in ("/", "/reviews", "/how-it-works"):
            assert b'href="/how-it-works"' in client.get(path).data


def test_toy_pipeline_upload_contains_complete_review_data():
    from test_pipeline import FakeLLM

    data = {
        "question_paper": (io.BytesIO("प्रश्न 1 (2 अंक)\nप्रश्न 2 (2 अंक)\nप्रश्न 3 (2 अंक)".encode()), "paper.txt"),
        "rubric": (io.BytesIO("ज्ञान: सही विचार, 2 अंक प्रति प्रश्न".encode()), "rubric.txt"),
        "model_answers_text": "1. सही उत्तर\n2. सही उत्तर\n3. सही उत्तर",
        "answer_files": (io.BytesIO("जल जीवन है\nपृथ्वी गोल है".encode()), "student.txt"),
        "provider": "ollama", "use_critic": "on",
    }
    with app.test_client() as client, patch("auto_assessment.pipeline.create_llm", return_value=FakeLLM()):
        response = client.post("/api/assess", data=data, buffered=True)
    events = [json.loads(line) for line in response.data.splitlines()]
    final = events[-1]
    assert final["type"] == "final"
    assert final["result"]["total"] == 2
    assert final["result"]["max_total"] == 6
    assert final["result"]["overall"]["summary"]
    assert final["result"]["overall"]["top_priorities"]
    assert "warnings" in final["result"]
    assert [item["question_id"] for item in final["items"]] == ["1", "2", "3"]
    assert final["items"][0]["answer"] == "जल जीवन है"
    assert final["items"][0]["model_answer"] == "सही उत्तर"
    assert not final["result"]["grades"][1]["attempted"]
    assert final["result"]["grades"][0]["criterion_scores"][0]["marks_withheld"] == 1


def test_question_progress_events_are_forwarded():
    def fake_stream(**kwargs):
        yield {"type": "progress", "node": "clean_align"}
        yield {"type": "question", "completed": 0, "total": 3, "question_id": "1"}
        yield {"type": "question", "completed": 3, "total": 3, "question_id": "3"}
        yield {
            "type": "final", "run_id": "test-run",
            "state": {"result": {"total": 1}, "report_md": "# Feedback",
                      "transcript": "First page", "items": []},
        }

    with app.test_client() as client, patch("app.stream_assessment", side_effect=fake_stream):
        response = client.post("/api/assess", data=materials(), buffered=True)
    events = [json.loads(line) for line in response.data.splitlines()]
    assert events[1] == {"type": "question", "completed": 0, "total": 3, "question_id": "1"}
    assert events[2] == {"type": "question", "completed": 3, "total": 3, "question_id": "3"}


def test_upload_stream_order_and_temporary_cleanup():
    saved_paths = []

    def fake_stream(**kwargs):
        assert kwargs["settings"].use_critic
        assert kwargs["settings"].max_grade_retries == 2
        assert kwargs["model_answers_text"] == "1. Reference answer"
        paths = [Path(path) for path in kwargs["answer_files"]]
        saved_paths.extend(paths)
        assert [path.read_text() for path in paths] == ["First page", "Second page"]
        assert paths[0] != paths[1]
        yield {"type": "progress", "node": "ingest", "update": {"private": "omit"}}
        assert all(path.exists() for path in paths)
        yield {
            "type": "final", "run_id": "test-run",
            "state": {"result": {"total": 1}, "report_md": "# Feedback",
                      "transcript": "First page", "items": []},
        }

    with app.test_client() as client, patch("app.stream_assessment", side_effect=fake_stream):
        response = client.post("/api/assess", data=materials(), buffered=True)
    assert response.status_code == 200
    assert response.mimetype == "application/x-ndjson"
    events = [json.loads(line) for line in response.data.splitlines()]
    assert events[0] == {"type": "progress", "node": "ingest"}
    assert events[1]["result"]["total"] == 1
    assert all(not path.exists() for path in saved_paths)


@pytest.mark.parametrize("change", [
    {"provider": "invalid"}, {"max_grade_retries": "5"},
    {"max_grade_retries": "oops"}, {"ollama_base_url": "file:///etc"},
    {"ollama_model": ""}, {"question_paper": None}, {"answer_files": []},
    {"model_answers_text": ""},
    {"model_answers": (b"answer", "answer.txt")},
    {"question_paper": (b"bad", "bad.html")},
])
def test_invalid_requests_stop_before_pipeline(change):
    data = materials()
    for key, value in change.items():
        if value is None:
            data.pop(key)
        else:
            data[key] = (io.BytesIO(value[0]), value[1]) if isinstance(value, tuple) else value
    with app.test_client() as client, patch("app.stream_assessment") as run:
        response = client.post("/api/assess", data=data)
    assert response.status_code == 400
    assert response.json["error"]
    run.assert_not_called()


def test_stream_failure_is_explicit_and_cleans_up():
    paths = []

    def broken_stream(**kwargs):
        paths.extend(Path(path) for path in kwargs["answer_files"])
        raise RuntimeError("Model unavailable")
        yield

    with app.test_client() as client, patch("app.stream_assessment", side_effect=broken_stream):
        response = client.post("/api/assess", data=materials(), buffered=True)
    assert json.loads(response.data)["error"] == "RuntimeError: Model unavailable"
    assert all(not path.exists() for path in paths)


def test_cross_origin_and_large_requests_rejected():
    with app.test_client() as client, patch("app.stream_assessment") as run:
        response = client.post("/api/assess", headers={"Origin": "https://elsewhere.test"})
        assert response.status_code == 403
        with patch.dict(app.config, MAX_CONTENT_LENGTH=10):
            response = client.post("/api/assess", data=materials())
            assert response.status_code == 413
            assert response.json["error"]
    run.assert_not_called()


def test_empty_file_and_duplicate_reference_rejected():
    with app.test_client() as client, patch("app.stream_assessment") as run:
        data = materials()
        data["question_paper"] = (io.BytesIO(b""), "paper.txt")
        response = client.post("/api/assess", data=data)
        assert response.status_code == 400
        assert "empty" in response.json["error"]
        data = materials()
        data["question_paper"] = [
            (io.BytesIO(b"first"), "first.txt"), (io.BytesIO(b"second"), "second.txt"),
        ]
        assert client.post("/api/assess", data=data).status_code == 400
    run.assert_not_called()


def test_disconnected_stream_removes_temporary_files():
    paths = []

    def fake_stream(**kwargs):
        paths.extend(Path(path) for path in kwargs["answer_files"])
        yield {"type": "progress", "node": "ingest"}
        yield {"type": "progress", "node": "parse_spec"}

    with app.test_client() as client, patch("app.stream_assessment", side_effect=fake_stream):
        response = client.post("/api/assess", data=materials(), buffered=False)
        assert paths and all(path.exists() for path in paths)
        response.close()
    assert all(not path.exists() for path in paths)
