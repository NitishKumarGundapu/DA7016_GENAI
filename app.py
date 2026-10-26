"""Gradio demo UI:  python app.py"""
import json
import tempfile
from pathlib import Path

import gradio as gr

from auto_assessment.config import Settings
from auto_assessment.pipeline import stream_assessment, tracing_enabled

NODE_LABELS = {
    "parse_spec": "Parsed question paper + rubric + model answers",
    "transcribe": "Transcribed handwritten pages (vision model)",
    "segment": "Mapped answers to questions",
    "grade_question": "Graded a question (grade -> verify -> retry loop)",
    "aggregate": "Totalled marks",
    "summarize": "Wrote overall feedback + report",
}
PROVIDERS = ["openai_then_ollama", "ollama_then_openai", "openai", "ollama"]


def _paths(files):
    if not files:
        return []
    files = files if isinstance(files, list) else [files]
    return [f if isinstance(f, str) else f.name for f in files]


def grade(qp, rubric, model_ans, spec, sheets, provider, openai_model, openai_vision,
          ollama_model, ollama_vision, ollama_url, use_critic, max_retries):
    log = []

    def snap(**kw):
        base = dict(status="\n".join(log), total="", table=[], report="", transcript="", json_file=None)
        base.update(kw)
        return (base["status"], base["total"], base["table"], base["report"],
                base["transcript"], base["json_file"])

    try:
        s = Settings.from_env()
        s.provider, s.openai_model, s.openai_vision_model = provider, openai_model, openai_vision
        s.ollama_model, s.ollama_vision_model, s.ollama_base_url = ollama_model, ollama_vision, ollama_url
        s.use_critic, s.max_grade_retries = use_critic, int(max_retries)

        log.append("Starting...")
        yield snap()
        spec_paths = _paths(spec)
        for ev in stream_assessment(
            question_paper=(_paths(qp) or [None])[0], rubric=(_paths(rubric) or [None])[0],
            model_answers=(_paths(model_ans) or [None])[0], spec_file=spec_paths[0] if spec_paths else None,
            answer_files=_paths(sheets), settings=s, tags=["gradio"],
        ):
            if ev["type"] == "progress":
                log.append(f"- {NODE_LABELS.get(ev['node'], ev['node'])}")
                yield snap()
            else:
                st = ev["state"]
                r = st["result"]
                table = [[g["question_id"], g["marks_awarded"], g["max_marks"],
                          "yes" if g["attempted"] else "no", "yes" if g["verified"] else "no"]
                         for g in r["grades"]]
                tmp = Path(tempfile.mkdtemp()) / "result.json"
                tmp.write_text(json.dumps(r, indent=2), encoding="utf-8")
                trace = (f"LangSmith run id `{ev['run_id']}`" if tracing_enabled()
                         else "LangSmith tracing is off (set LANGSMITH_TRACING / LANGSMITH_API_KEY)")
                log.append(f"Done. {trace}")
                yield snap(total=f"## Total: {r['total']} / {r['max_total']}  ({r['percentage']}%)",
                           table=table, report=st["report_md"], transcript=st["transcript"],
                           json_file=str(tmp))
    except Exception as e:  # surface errors in the UI instead of a blank screen
        log.append(f"**Error:** {type(e).__name__}: {e}")
        yield snap()


with gr.Blocks(title="Auto-Assessment") as demo:
    gr.Markdown("# Auto-Assessment\nGrade a handwritten answer sheet against a rubric, with detailed feedback.")
    with gr.Row():
        with gr.Column():
            qp = gr.File(label="Question paper (.pdf/.txt/.md)", type="filepath")
            rubric = gr.File(label="Rubric (.pdf/.txt/.md)", type="filepath")
            model_ans = gr.File(label="Model answers (.pdf/.txt/.md)", type="filepath")
            spec = gr.File(label="(Optional) structured spec .json - replaces the three above", type="filepath")
            sheets = gr.File(label="Student answer sheet - scans (png/jpg/pdf)", file_count="multiple",
                             type="filepath")
            with gr.Accordion("Model & pipeline settings", open=False):
                d = Settings.from_env()
                provider = gr.Dropdown(PROVIDERS, value=d.provider, label="LLM provider mode")
                openai_model = gr.Textbox(d.openai_model, label="OpenAI text model")
                openai_vision = gr.Textbox(d.openai_vision_model, label="OpenAI vision model")
                ollama_model = gr.Textbox(d.ollama_model, label="Ollama text model")
                ollama_vision = gr.Textbox(d.ollama_vision_model, label="Ollama vision model")
                ollama_url = gr.Textbox(d.ollama_base_url, label="Ollama base URL")
                use_critic = gr.Checkbox(d.use_critic, label="LLM critic verification")
                max_retries = gr.Slider(0, 4, value=d.max_grade_retries, step=1, label="Max re-grade retries")
            go = gr.Button("Grade", variant="primary")
        with gr.Column():
            status = gr.Markdown()
            total = gr.Markdown()
            table = gr.Dataframe(headers=["Q", "Marks", "Max", "Attempted", "Verified"], interactive=False)
            json_file = gr.File(label="Download result.json")
    with gr.Tabs():
        with gr.Tab("Feedback report"):
            report = gr.Markdown()
        with gr.Tab("Transcript"):
            transcript = gr.Textbox(lines=20, label="Transcribed answer sheet", interactive=False)

    go.click(grade,
             [qp, rubric, model_ans, spec, sheets, provider, openai_model, openai_vision,
              ollama_model, ollama_vision, ollama_url, use_critic, max_retries],
             [status, total, table, report, transcript, json_file])

if __name__ == "__main__":
    demo.launch()
