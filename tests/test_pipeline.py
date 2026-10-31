"""Offline regression checks: python -m pytest tests/test_pipeline.py."""
import base64
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from auto_assessment.config import Settings
from auto_assessment.cli import main as cli_main
from auto_assessment.graph import build_graph
from auto_assessment.ingest import extract_document, IMAGE_MIME_TYPES
from auto_assessment.llm_client import LLMClient
from auto_assessment.pipeline import run_assessment, stream_assessment
from auto_assessment.models import (
    AnswerSegment, Criterion, CriterionScore, CriticVerdict, ExamSpec, GradeDraft,
    Question, ReportFeedback, ReportLabels, Segmentation,
)
from auto_assessment.verify import build_grade, check_draft, evidence_found, norm_qid, normalize_spec


def spec():
    return ExamSpec(title="परीक्षा", questions=[
        Question(id=str(i), text=f"प्रश्न {i}", max_marks=2, model_answer="सही उत्तर",
                 criteria=[Criterion(name="ज्ञान", description="सही विचार", max_marks=2)])
        for i in range(1, 4)
    ])


def draft(answer):
    return GradeDraft(
        criterion_scores=[CriterionScore(criterion="ज्ञान", marks_awarded=1,
                                         evidence=answer, justification="एक अंक मिला, एक अंक अधूरा")],
        strengths=answer, gaps="एक विचार अधूरा, एक अंक नहीं मिला",
        how_to_improve="दोनों विचार लिखें",
    )


def reference_inputs(directory):
    paths = {}
    for role, text in {
        "question_paper": "प्रश्न 1 (2 अंक)\nप्रश्न 2 (2 अंक)\nप्रश्न 3 (2 अंक)",
        "rubric": "ज्ञान: सही विचार, 2 अंक प्रति प्रश्न",
        "model_answers": "1. सही उत्तर\n2. सही उत्तर\n3. सही उत्तर",
    }.items():
        path = directory / f"{role}.txt"
        path.write_text(text, encoding="utf-8")
        paths[role] = str(path)
    return paths


def ocr_response(text, finish_reason="stop"):
    return Mock(json=Mock(return_value={
        "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
    }))


class FakeLLM:
    def __init__(self):
        labels = {key: "शीर्षक" for key in ReportLabels.model_fields}
        labels.update(unattempted="उत्तर नहीं दिया, शून्य अंक", attempt_advice="प्रश्न हल करें")
        self.responses = {
            ExamSpec: [spec()],
            Segmentation: [Segmentation(answers=[
                AnswerSegment(question_id="1", attempted=True, answer_text="जल जीवन है"),
                AnswerSegment(question_id="2", attempted=False, answer_text=""),
                AnswerSegment(question_id="3", attempted=True, answer_text="पृथ्वी गोल है"),
            ])],
            GradeDraft: [draft("जल जीवन है"), draft("पृथ्वी गोल है")],
            CriticVerdict: [CriticVerdict(ok=True, issues=[]), CriticVerdict(ok=True, issues=[])],
            ReportFeedback: [ReportFeedback(summary="अभ्यास करें", top_priorities=["प्रश्न 1 और 3"],
                                             labels=ReportLabels(**labels), diagnostic_translations=[])],
        }
        self.calls = []

    def describe(self):
        return {"provider": "offline-test"}

    def structured(self, schema, **options):
        owner = self

        class Chain:
            def invoke(self, messages):
                owner.calls.append((schema, messages, options))
                return owner.responses[schema].pop(0)

            def batch(self, messages, config):
                return [self.invoke(message) for message in messages]

        return Chain()


class FailFirstGradeDraft(FakeLLM):
    def __init__(self):
        super().__init__()
        self.failed = False

    def structured(self, schema, **options):
        chain = super().structured(schema, **options)
        if schema is not GradeDraft:
            return chain
        owner = self

        class Chain:
            def invoke(self, messages):
                result = chain.invoke(messages)
                if not owner.failed:
                    owner.failed = True
                    raise RuntimeError("simulated grading failure")
                return result

        return Chain()


def test_failed_question_is_skipped_and_pipeline_continues(caplog):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        answer = directory / "student.txt"
        answer.write_text("जल जीवन है\nपृथ्वी गोल है", encoding="utf-8")
        llm = FailFirstGradeDraft()
        llm.responses[ReportFeedback][0].diagnostic_translations = [
            "प्रश्न 1: मूल्यांकन विफल, समीक्षा आवश्यक",
            "प्रश्न 1: खुली समीक्षा टिप्पणियाँ",
            "मूल्यांकन विफल; अंक रोके गए",
        ]
        state = build_graph(llm, Settings(use_critic=True)).invoke(
            {**reference_inputs(directory), "answer_files": [str(answer)]})
        assert "Skipping question=1" in caplog.text
        grades = state["result"]["grades"]
        assert [g["question_id"] for g in grades] == ["1", "2", "3"]
        assert grades[0]["attempted"] and not grades[0]["verified"]
        assert grades[0]["marks_awarded"] == 0
        assert grades[0]["attempts"] == 0
        assert grades[1]["attempts"] == 0 and not grades[1]["attempted"]
        assert grades[2]["verified"] and grades[2]["marks_awarded"] == 1
        assert state["result"]["total"] == 1
        assert any("Q1: grading failed and the question was skipped" in w
                   for w in state["warnings"])
        assert "मूल्यांकन विफल; अंक रोके गए" in state["report_md"]


def test_full_pipeline_and_streaming(caplog):
    for streaming in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            answer = directory / "student.txt"
            answer.write_text("जल जीवन है\nपृथ्वी गोल है", encoding="utf-8")
            llm = FakeLLM()
            kwargs = dict(**reference_inputs(directory), answer_files=[answer],
                          settings=Settings(use_critic=True))
            with patch("auto_assessment.pipeline.create_llm", return_value=llm):
                if streaming:
                    events = list(stream_assessment(**kwargs))
                    state = events[-1]["state"]
                    assert [e["node"] for e in events[:-1] if e["type"] == "progress"] == [
                        "ingest", "parse_spec", "clean_align", "grade_question", "aggregate", "report",
                    ]
                    question_events = [e for e in events if e["type"] == "question"]
                    assert question_events
                    assert question_events[-1]["completed"] == question_events[-1]["total"] == 3
                    assert [e["question_id"] for e in question_events] == ["1", "1", "2", "2", "3", "3"]
                else:
                    state = run_assessment(**kwargs)
            assert "Starting node=ingest" in caplog.text
            assert "Finished node=report" in caplog.text
            assert "Starting node=grade question=1 attempt=1" in caplog.text
            assert state["result"]["total"] == 2
            assert state["result"]["max_total"] == 6
            grades = state["result"]["grades"]
            assert [g["question_id"] for g in grades] == ["1", "2", "3"]
            assert grades[1]["attempts"] == 0 and not grades[1]["attempted"]
            assert grades[1]["gaps"] == "उत्तर नहीं दिया, शून्य अंक"
            assert grades[0]["criterion_scores"][0]["marks_withheld"] == 1
            assert all(g["verified"] for g in grades)
            assert set(state["items"][0]) == {
                "question_id", "question", "answer", "model_answer", "rubric", "max_marks",
            }
            grading_calls = [c for c in llm.calls if c[0] is GradeDraft]
            assert [json.loads(c[1][-1].content.split("\n", 1)[1])["question_id"]
                    for c in grading_calls] == ["1", "3"]
            assert next(c for c in llm.calls if c[0] is ExamSpec)[2]["reasoning"] is False
            assert next(c for c in llm.calls if c[0] is Segmentation)[2]["reasoning"] is False
            assert all("reasoning" not in c[2] for c in grading_calls)
            assert "उत्तर नहीं दिया" in state["report_md"]
            assert "Assessment Report" not in state["report_md"]
            report_input = next(c[1][-1].content for c in llm.calls if c[0] is ReportFeedback)
            assert "marks_withheld" in report_input and "criterion_scores" in report_input


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("inline", [False, True])
def test_optional_rubric_uses_whole_question_criterion(streaming, inline):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        inputs = reference_inputs(directory)
        inputs.pop("rubric")
        if inline:
            inputs["model_answers_text"] = Path(inputs.pop("model_answers")).read_text(encoding="utf-8")
        answer = directory / "student.txt"
        answer.write_text("जल जीवन है\nपृथ्वी गोल है", encoding="utf-8")
        llm = FakeLLM()
        for grading in llm.responses[GradeDraft]:
            grading.criterion_scores[0].criterion = "Whole question"
        # Ignore any model-proposed allocations when no rubric was supplied.
        llm.responses[ExamSpec][0].questions[0].criteria = []
        llm.responses[ExamSpec][0].questions[1].criteria[0].max_marks = 999
        with patch("auto_assessment.pipeline.create_llm", return_value=llm):
            kwargs = dict(**inputs, answer_files=[answer], settings=Settings(use_critic=True))
            state = (list(stream_assessment(**kwargs))[-1]["state"] if streaming
                     else run_assessment(**kwargs))
        assert state["result"]["total"] == 2
        assert state["result"]["max_total"] == 6
        assert all(g["verified"] for g in state["result"]["grades"])
        for question in state["spec"]["questions"]:
            assert len(question["criteria"]) == 1
            criterion = question["criteria"][0]
            assert criterion["name"] == "Whole question"
            assert criterion["max_marks"] == question["max_marks"]
            assert question["model_answer"] in criterion["description"]
        parse_call = next(c for c in llm.calls if c[0] is ExamSpec)
        assert "RUBRIC:\n\n" in parse_call[1][-1].content


def test_rubric_question_without_explicit_allocations_uses_reported_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        inputs = reference_inputs(directory)
        answer = directory / "student.txt"
        answer.write_text("जल जीवन है\nपृथ्वी गोल है", encoding="utf-8")
        llm = FakeLLM()
        llm.responses[ExamSpec][0].questions[0].criteria = []
        llm.responses[GradeDraft][0].criterion_scores[0].criterion = "Whole question"
        llm.responses[ReportFeedback][0].diagnostic_translations = [
            "Q1: no explicit rubric allocation; graded against the model answer."
        ]
        with patch("auto_assessment.pipeline.create_llm", return_value=llm):
            state = run_assessment(
                **inputs, answer_files=[answer], settings=Settings(use_critic=True)
            )
        criterion = state["spec"]["questions"][0]["criteria"][0]
        assert criterion["name"] == "Whole question"
        assert criterion["max_marks"] == state["spec"]["questions"][0]["max_marks"]
        assert state["result"]["warnings"] == [
            "Q1: no explicit rubric allocation; graded against the model answer."
        ]
        assert "Q1: no explicit rubric allocation" in state["report_md"]


@pytest.mark.parametrize("missing", ["question_paper", "model_answers", "answer_files"])
@pytest.mark.parametrize("streaming", [False, True])
def test_required_documents_stop_before_model_creation(missing, streaming):
    inputs = dict(question_paper="paper.txt", model_answers="model.txt", answer_files=["student.txt"])
    inputs[missing] = [] if missing == "answer_files" else None
    with patch("auto_assessment.pipeline.create_llm") as llm:
        with pytest.raises(ValueError, match="Provide"):
            if streaming:
                list(stream_assessment(**inputs))
            else:
                run_assessment(**inputs)
        llm.assert_not_called()


@pytest.mark.parametrize("max_marks", [0, float("nan")])
def test_no_rubric_still_requires_valid_question_marks(max_marks):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        inputs = reference_inputs(directory)
        inputs.pop("rubric")
        answer = directory / "student.txt"
        answer.write_text("जल जीवन है", encoding="utf-8")
        llm = FakeLLM()
        llm.responses[ExamSpec][0].questions[0].max_marks = max_marks
        with patch("auto_assessment.pipeline.create_llm", return_value=llm):
            with pytest.raises(ValueError, match="maximum marks"):
                run_assessment(**inputs, answer_files=[answer], settings=Settings(use_critic=False))
        assert not any(c[0] is GradeDraft for c in llm.calls)


def test_four_sources_mixed_scans_and_inline_correct_answers():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        question = directory / "paper.txt"
        question.write_text("प्रश्न 1\nप्रश्न 2\nप्रश्न 3", encoding="utf-8")
        rubric = directory / "rubric.png"
        sheet = directory / "student.png"
        rubric.write_bytes(b"rubric-image")
        sheet.write_bytes(b"student-image")
        typed = directory / "student.txt"
        typed.write_text("जल जीवन है", encoding="utf-8")
        llm = FakeLLM()
        with patch("auto_assessment.pipeline.create_llm", return_value=llm), \
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
                patch("auto_assessment.ingest.requests.post", side_effect=[
                    ocr_response("ज्ञान: 2 अंक प्रति प्रश्न"), ocr_response("पृथ्वी गोल है"),
                ]) as post:
            state = run_assessment(question_paper=question, rubric=rubric,
                                   model_answers_text="1. सही उत्तर\n2. सही उत्तर\n3. सही उत्तर",
                                   answer_files=[typed, sheet], settings=Settings(use_critic=False))
        assert state["document_count"] == 2
        assert state["transcript"].index("जल जीवन है") < state["transcript"].index("पृथ्वी गोल है")
        assert state["rubric"].endswith("ज्ञान: 2 अंक प्रति प्रश्न")
        assert post.call_count == 2


@pytest.mark.parametrize("extension", [".pdf", *IMAGE_MIME_TYPES])
def test_documents_sent_unchanged_to_openrouter(extension):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"document{extension}"
        raw = b"original-file-bytes"
        path.write_bytes(raw)
        response = ocr_response("पहला पृष्ठ\n\nदूसरा पृष्ठ")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
                patch("auto_assessment.ingest.requests.post", return_value=response) as post:
            assert extract_document(str(path), Settings()) == "पहला पृष्ठ\n\nदूसरा पृष्ठ"
        response.raise_for_status.assert_called_once_with()
        assert post.call_args.args == ("https://openrouter.ai/api/v1/chat/completions",)
        options = post.call_args.kwargs
        assert options["headers"]["Authorization"] == "Bearer test-key"
        assert options["timeout"] == (10, 180)
        payload = options["json"]
        assert payload["model"] == "mistralai/mistral-small-3.1-24b-instruct"
        assert payload["temperature"] == 0
        assert payload["reasoning"] == {"enabled": False}
        assert "Preserve the original language" in payload["messages"][0]["content"]
        assert "Diagrams/graphs/tables/figures" in payload["messages"][0]["content"]
        assert "axis labels" in payload["messages"][0]["content"]
        assert "Convert diagrams" in payload["messages"][1]["content"][0]["text"]
        attachment = payload["messages"][1]["content"][1]
        if extension == ".pdf":
            assert attachment["type"] == "file"
            assert attachment["file"]["filename"] == path.name
            data_url = attachment["file"]["file_data"]
            assert data_url.startswith("data:application/pdf;base64,")
            assert payload["plugins"] == [{"id": "file-parser", "pdf": {"engine": "mistral-ocr"}}]
        else:
            assert attachment["type"] == "image_url"
            data_url = attachment["image_url"]["url"]
            assert data_url.startswith(f"data:{IMAGE_MIME_TYPES[extension]};base64,")
            assert "plugins" not in payload
        assert base64.b64decode(data_url.split(",", 1)[1]) == raw


def test_text_read_without_ocr_and_custom_vision_model():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "paper.md"
        path.write_text("प्रश्न 1\n", encoding="utf-8")
        with patch("auto_assessment.ingest.requests.post") as post:
            assert extract_document(str(path), Settings()) == "प्रश्न 1\n"
            post.assert_not_called()
        path = Path(tmp) / "paper.pdf"
        path.write_bytes(b"original-pdf")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
                patch("auto_assessment.ingest.requests.post", return_value=ocr_response("प्रश्न 1")) as post:
            extract_document(str(path), Settings(openrouter_vision_model="custom/vision"))
            assert post.call_args.kwargs["json"]["model"] == "custom/vision"
            assert post.call_args.kwargs["json"]["reasoning"] == {"enabled": False}
            extract_document(str(path), Settings(
                openrouter_vision_model="custom/vision", extraction_reasoning=True))
        assert post.call_args.kwargs["json"]["reasoning"] == {"enabled": True}


@pytest.mark.parametrize("body", [
    {"error": {"message": "OCR provider unavailable"}},
    {}, {"choices": []}, {"choices": [{"message": {}}]}, [],
])
def test_ocr_rejects_api_errors_and_malformed_responses(body):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "paper.pdf"
        path.write_bytes(b"original-pdf")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
                patch("auto_assessment.ingest.requests.post", return_value=Mock(json=Mock(return_value=body))):
            with pytest.raises((ValueError, RuntimeError), match="OCR"):
                extract_document(str(path), Settings())


@pytest.mark.parametrize("text,reason", [
    ("", "stop"), ("   ", "stop"), (None, "stop"),
    ("partial transcript", "length"), ("filtered", "content_filter"),
])
def test_ocr_rejects_empty_or_incomplete_transcriptions(text, reason):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "paper.pdf"
        path.write_bytes(b"original-pdf")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
                patch("auto_assessment.ingest.requests.post", return_value=ocr_response(text, reason)):
            with pytest.raises(ValueError, match="no readable text|did not complete"):
                extract_document(str(path), Settings())


def test_ocr_requires_key_and_rejects_unsupported_or_empty_files():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "paper.pdf"
        path.write_bytes(b"original-pdf")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}), \
                patch("auto_assessment.ingest.requests.post") as post:
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                extract_document(str(path), Settings())
            post.assert_not_called()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
                patch("auto_assessment.ingest.requests.post") as post:
            for name in ("empty.txt", "empty.pdf", "paper.tiff"):
                path = Path(tmp) / name
                path.write_bytes(b"")
                with pytest.raises(ValueError, match="no readable text|empty|Unsupported"):
                    extract_document(str(path), Settings())
            post.assert_not_called()


@pytest.mark.parametrize("error", [requests.Timeout("OCR timeout"), requests.HTTPError("OCR HTTP error")])
def test_ocr_transport_errors_propagate(error):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "paper.pdf"
        path.write_bytes(b"original-pdf")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
                patch("auto_assessment.ingest.requests.post") as post:
            if isinstance(error, requests.Timeout):
                post.side_effect = error
            else:
                post.return_value.raise_for_status.side_effect = error
            with pytest.raises(type(error)):
                extract_document(str(path), Settings())


@pytest.mark.parametrize("answer", ["जल जीवन है", "தமிழ் விடை", "中文答案", "إجابة صحيحة"])
def test_unicode_evidence(answer):
    assert evidence_found(answer, f"1. {answer}")
    assert not evidence_found(answer, "unrelated English content")
    assert norm_qid(answer)
    assert not evidence_found("...", answer)


@pytest.mark.parametrize("case", ["missing", "inconsistent", "zero", "nan", "duplicate"])
def test_require_explicit_valid_rubric_marks(case):
    paper = spec()
    q = paper.questions[0]
    if case == "missing":
        q.criteria = []
    elif case == "inconsistent":
        q.criteria[0].max_marks = 1
    elif case == "zero":
        q.criteria[0].max_marks = 0
    elif case == "nan":
        q.max_marks = float("nan")
    else:
        paper.questions[1].id = q.id
    with pytest.raises(ValueError):
        normalize_spec(paper)


def test_bad_marks_stop_before_grading():
    llm = FakeLLM()
    paper = spec().model_dump()
    paper["questions"][0]["criteria"][0]["max_marks"] = 1
    llm.responses[ExamSpec] = [ExamSpec.model_validate(paper)]
    with tempfile.TemporaryDirectory() as tmp:
        answer = Path(tmp) / "student.txt"
        answer.write_text("जल जीवन है\nपृथ्वी गोल है", encoding="utf-8")
        with pytest.raises(ValueError, match="criterion marks sum"):
            build_graph(llm, Settings(use_critic=False)).invoke(
                {**reference_inputs(Path(tmp)), "answer_files": [str(answer)]})
    assert not any(c[0] is GradeDraft for c in llm.calls)


def test_missing_alignment_is_skipped_and_marked_for_review():
    llm = FakeLLM()
    llm.responses[Segmentation][0].answers.pop()
    llm.responses[ReportFeedback][0].diagnostic_translations = [
        "प्रश्न 3: संरेखण विफल, समीक्षा आवश्यक",
        "प्रश्न 3: खुली समीक्षा टिप्पणियाँ",
        "कोई संरेखित उत्तर नहीं मिला; अंक रोके गए",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        answer = Path(tmp) / "student.txt"
        answer.write_text("जल जीवन है\nपृथ्वी गोल है", encoding="utf-8")
        state = build_graph(llm, Settings(use_critic=False)).invoke(
            {**reference_inputs(Path(tmp)), "answer_files": [str(answer)]})
    grades = state["result"]["grades"]
    assert [g["question_id"] for g in grades] == ["1", "2", "3"]
    assert grades[2]["verified"] is False and grades[2]["marks_awarded"] == 0
    assert grades[2]["attempted"] is False
    assert any("Q3: answer alignment failed" in w for w in state["warnings"])
    assert [json.loads(c[1][-1].content.split("\n", 1)[1])["question_id"]
            for c in llm.calls if c[0] is GradeDraft] == ["1"]
    assert "कोई संरेखित उत्तर नहीं मिला; अंक रोके गए" in state["report_md"]


def test_unverified_draft_cannot_double_count_or_award_ungrounded_marks():
    question = spec().questions[0]
    grading = draft("not in the answer")
    grading.criterion_scores *= 2
    issues = check_draft(question, "जल जीवन है", grading)
    result = build_grade(question, "जल जीवन है", grading, attempts=1, issues=issues)
    assert not result.verified and result.marks_awarded == 0
    assert len(result.criterion_scores) == 1
    assert result.criterion_scores[0].marks_withheld == 2


def test_structured_grading_model_preserves_temperature():
    llm = object.__new__(LLMClient)
    with patch.object(llm, "chat") as chat:
        llm.structured(ExamSpec, temperature=0)
    chat.assert_called_once_with(temperature=0, reasoning=None)
    chat.return_value.with_structured_output.assert_called_once_with(ExamSpec)


@pytest.mark.parametrize("extension", [".png", ".pdf"])
def test_all_four_document_roles_use_openrouter_ocr(extension):
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / f"{role}{extension}" for role in ("paper", "rubric", "model", "student")]
        for path in paths:
            path.write_bytes(b"original-file-bytes")
        llm = FakeLLM()
        responses = [
            ocr_response(text)
            for text in ("प्रश्न 1, 2, 3", "ज्ञान: 2 अंक", "सही उत्तर", "जल जीवन है\nपृथ्वी गोल है")
        ]
        with patch("auto_assessment.pipeline.create_llm", return_value=llm), \
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
                patch("auto_assessment.ingest.requests.post", side_effect=responses) as post:
            state = run_assessment(question_paper=paths[0], rubric=paths[1], model_answers=paths[2],
                                   answer_files=[paths[3]], settings=Settings(use_critic=False))
        assert state["result"]["total"] == 2
        assert post.call_count == 4


def test_answer_can_span_pages_and_invalid_grading_retries():
    with tempfile.TemporaryDirectory() as tmp:
        pages = [Path(tmp) / f"page{i}.txt" for i in range(2)]
        pages[0].write_text("जल जीवन", encoding="utf-8")
        pages[1].write_text("है\nपृथ्वी गोल है", encoding="utf-8")
        llm = FakeLLM()
        llm.responses[GradeDraft].insert(0, draft("invented evidence"))
        state = build_graph(llm, Settings(use_critic=False)).invoke(
            {**reference_inputs(Path(tmp)), "answer_files": [str(p) for p in pages]})
        assert state["result"]["grades"][0]["attempts"] == 2
        assert state["result"]["grades"][0]["verified"]
        calls = [c[1][-1].content for c in llm.calls if c[0] is GradeDraft]
        assert "YOUR PREVIOUS ATTEMPT WAS REJECTED" in calls[1]


def test_hallucinated_alignment_is_skipped_not_graded():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "student.txt"
        path.write_text("जल जीवन है\nपृथ्वी गोल है", encoding="utf-8")
        llm = FakeLLM()
        llm.responses[Segmentation][0].answers[0].answer_text = "invented answer"
        llm.responses[ReportFeedback][0].diagnostic_translations = [
            "प्रश्न 1: संरेखण विफल, समीक्षा आवश्यक",
            "प्रश्न 1: खुली समीक्षा टिप्पणियाँ",
            "उत्तर प्रतिलेख में नहीं मिला; अंक रोके गए",
        ]
        state = build_graph(llm, Settings(use_critic=False)).invoke(
            {**reference_inputs(Path(tmp)), "answer_files": [str(path)]})
    grades = state["result"]["grades"]
    assert [g["question_id"] for g in grades] == ["1", "2", "3"]
    assert grades[0]["attempted"] and not grades[0]["verified"]
    assert grades[0]["marks_awarded"] == 0
    assert grades[2]["verified"] and grades[2]["marks_awarded"] == 1
    grading_prompts = [c[1][-1].content for c in llm.calls if c[0] is GradeDraft]
    assert grading_prompts
    assert all('"question_id": "3"' in prompt for prompt in grading_prompts)
    assert any("Q1: answer alignment failed" in w for w in state["warnings"])
    assert "उत्तर प्रतिलेख में नहीं मिला; अंक रोके गए" in state["report_md"]


def test_uncertain_transcription_is_unverified_and_diagnostics_are_localized():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "student.txt"
        path.write_text("जल जीवन है [?]\nपृथ्वी गोल है", encoding="utf-8")
        llm = FakeLLM()
        llm.responses[Segmentation][0].answers[0].answer_text = "जल जीवन है [?]"
        llm.responses[ReportFeedback][0].diagnostic_translations = [
            "प्रतिलेखन अस्पष्ट: समीक्षा आवश्यक",
            "प्रश्न 1: समीक्षा आवश्यक", "अस्पष्ट शब्द: समीक्षा आवश्यक",
        ]
        state = build_graph(llm, Settings(use_critic=False, max_grade_retries=0)).invoke(
            {**reference_inputs(Path(tmp)), "answer_files": [str(path)]})
        assert not state["result"]["grades"][0]["verified"]
        assert state["result"]["warnings"] == [
            "प्रतिलेखन अस्पष्ट: समीक्षा आवश्यक", "प्रश्न 1: समीक्षा आवश्यक",
        ]
        assert "अस्पष्ट शब्द: समीक्षा आवश्यक" in state["report_md"]


@pytest.mark.parametrize("with_rubric", [False, True])
def test_cli_persists_report_aligned_dictionaries_and_marks(with_rubric):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        inputs = reference_inputs(directory)
        student = directory / "student.txt"
        student.write_text("जल जीवन है\nपृथ्वी गोल है", encoding="utf-8")
        output = directory / "output"
        args = ["auto-assessment", "--question-paper", inputs["question_paper"],
                "--model-answers", inputs["model_answers"],
                "--answers", str(student),
                "--no-critic", "--out", str(output)]
        llm = FakeLLM()
        if with_rubric:
            args.extend(["--rubric", inputs["rubric"]])
        else:
            for grading in llm.responses[GradeDraft]:
                grading.criterion_scores[0].criterion = "Whole question"
        with patch("sys.argv", args), patch("auto_assessment.pipeline.create_llm", return_value=llm):
            cli_main()
        assert {p.name for p in output.iterdir()} == {
            "report.md", "questions.json", "result.json", "transcript.txt",
        }
        assert json.loads((output / "result.json").read_text(encoding="utf-8"))["total"] == 2
        assert json.loads((output / "questions.json").read_text(encoding="utf-8"))[0]["answer"] == "जल जीवन है"
        assert "उत्तर नहीं दिया" in (output / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("args", [
    ["--answers", "student.txt", "--model-answers", "model.txt"],
    ["--answers", "student.txt", "--question-paper", "paper.txt"],
    ["--question-paper", "paper.txt", "--model-answers", "model.txt"],
    ["--question-paper", "paper.txt", "--model-answers", "model.txt",
     "--answers", "student.txt", "--spec", "spec.json"],
])
def test_cli_rejects_missing_required_inputs_and_removed_spec(args):
    with patch("sys.argv", ["auto-assessment", *args]), patch("auto_assessment.cli.run_assessment") as run:
        with pytest.raises(SystemExit) as exc:
            cli_main()
        assert exc.value.code == 2
        run.assert_not_called()
