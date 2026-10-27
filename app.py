"""Streamlit UI: run with ``streamlit run app.py``."""
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from auto_assessment.config import Settings
from auto_assessment.pipeline import stream_assessment, tracing_enabled

NODE_LABELS = {
    "parse_spec": "Parsed question paper, rubric, and model answers",
    "transcribe": "Transcribed handwritten pages",
    "segment": "Mapped answers to questions",
    "grade_question": "Graded and verified a question",
    "aggregate": "Totalled marks",
    "summarize": "Wrote feedback and report",
}
PROVIDERS = ["openai_then_ollama", "ollama_then_openai", "openai", "ollama"]


def _save_upload(upload, directory: Path) -> str | None:
    if upload is None:
        return None
    path = directory / Path(upload.name).name
    path.write_bytes(upload.getvalue())
    return str(path)


def main() -> None:
    st.set_page_config(page_title="Auto-Assessment", page_icon="📝", layout="wide")
    st.title("Auto-Assessment")
    st.write("Grade a handwritten answer sheet against a rubric, with detailed feedback.")

    defaults = Settings.from_env()
    left, right = st.columns(2)
    with left:
        question_paper = st.file_uploader("Question paper (.pdf/.txt/.md)", type=["pdf", "txt", "md"], key="qp")
        rubric = st.file_uploader("Rubric (.pdf/.txt/.md)", type=["pdf", "txt", "md"], key="rubric")
        model_answers = st.file_uploader("Model answers (.pdf/.txt/.md)", type=["pdf", "txt", "md"], key="model_answers")
        spec = st.file_uploader("Optional structured spec (.json); replaces the three files above", type=["json"], key="spec")
        sheets = st.file_uploader("Student answer sheet scans (.png/.jpg/.pdf)", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True, key="sheets")
        with st.expander("Model & pipeline settings"):
            provider = st.selectbox("LLM provider mode", PROVIDERS, index=PROVIDERS.index(defaults.provider) if defaults.provider in PROVIDERS else 0)
            openai_model = st.text_input("OpenAI text model", defaults.openai_model)
            openai_vision = st.text_input("OpenAI vision model", defaults.openai_vision_model)
            ollama_model = st.text_input("Ollama text model", defaults.ollama_model)
            ollama_vision = st.text_input("Ollama vision model", defaults.ollama_vision_model)
            ollama_url = st.text_input("Ollama base URL", defaults.ollama_base_url)
            use_critic = st.checkbox("LLM critic verification", defaults.use_critic)
            max_retries = st.slider("Max re-grade retries", 0, 4, defaults.max_grade_retries)
        run = st.button("Grade", type="primary", use_container_width=True)

    with right:
        status_area = st.empty()
        total_area = st.empty()
        results_area = st.empty()
        download_area = st.empty()
        report_area = st.empty()
        transcript_area = st.empty()

    if not run:
        return
    if not sheets:
        st.error("Upload at least one answer-sheet file.")
        return
    if not spec and not all([question_paper, rubric, model_answers]):
        st.error("Upload a JSON spec, or provide the question paper, rubric, and model answers.")
        return

    settings = Settings.from_env()
    settings.provider = provider
    settings.openai_model, settings.openai_vision_model = openai_model, openai_vision
    settings.ollama_model, settings.ollama_vision_model = ollama_model, ollama_vision
    settings.ollama_base_url = ollama_url
    settings.use_critic, settings.max_grade_retries = use_critic, max_retries

    logs: list[str] = ["Starting…"]
    status_area.markdown("\n".join(f"- {line}" for line in logs))
    try:
        with tempfile.TemporaryDirectory(prefix="auto-assessment-") as temp_dir:
            directory = Path(temp_dir)
            qp_path = _save_upload(question_paper, directory)
            rubric_path = _save_upload(rubric, directory)
            answers_path = _save_upload(model_answers, directory)
            spec_path = _save_upload(spec, directory)
            sheet_paths = [_save_upload(item, directory) for item in sheets]

            for event in stream_assessment(
                question_paper=qp_path,
                rubric=rubric_path,
                model_answers=answers_path,
                spec_file=spec_path,
                answer_files=sheet_paths,
                settings=settings,
                tags=["streamlit"],
            ):
                if event["type"] == "progress":
                    logs.append(NODE_LABELS.get(event["node"], event["node"]))
                    status_area.markdown("\n".join(f"- {line}" for line in logs))
                    continue

                state = event["state"]
                result = state["result"]
                rows = [{
                    "Question": grade["question_id"],
                    "Marks": grade["marks_awarded"],
                    "Max": grade["max_marks"],
                    "Attempted": "Yes" if grade["attempted"] else "No",
                    "Verified": "Yes" if grade["verified"] else "No",
                } for grade in result["grades"]]
                logs.append("Done. " + (f"LangSmith run id `{event['run_id']}`" if tracing_enabled() else "LangSmith tracing is off (set LANGSMITH_TRACING / LANGSMITH_API_KEY)."))
                status_area.markdown("\n".join(f"- {line}" for line in logs))
                total_area.markdown(f"## Total: {result['total']} / {result['max_total']} ({result['percentage']}%)")
                results_area.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                download_area.download_button(
                    "Download result.json",
                    data=json.dumps(result, indent=2).encode("utf-8"),
                    file_name="result.json",
                    mime="application/json",
                )
                with report_area.container():
                    st.subheader("Feedback report")
                    st.markdown(state["report_md"])
                with transcript_area.container():
                    st.subheader("Transcript")
                    st.text_area("Transcribed answer sheet", state["transcript"], height=300, disabled=True, label_visibility="collapsed")
    except Exception as exc:
        logs.append(f"Error: {type(exc).__name__}: {exc}")
        status_area.markdown("\n".join(f"- {line}" for line in logs))
        st.error(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
