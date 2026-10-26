"""Public entry points: run the graph (blocking) or stream progress events (for the UI)."""
from __future__ import annotations

import os
import uuid
from typing import Iterator

from .config import Settings
from .graph import build_graph
from .ingest import load_spec_json, read_text_document
from .llm import HybridLLM


def tracing_enabled() -> bool:
    flag = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or ""
    return flag.lower() in {"1", "true", "yes"} and bool(
        os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    )


def _prepare(question_paper, rubric, model_answers, spec_file, answer_files, settings, tags):
    if not answer_files:
        raise ValueError("Provide at least one answer-sheet file (image / PDF / .txt).")
    settings = settings or Settings.from_env()
    llm = HybridLLM(settings)
    inputs: dict = {"answer_files": [str(p) for p in answer_files], "warnings": [], "grades": []}

    spec_json = load_spec_json(spec_file) if spec_file else None
    if spec_json:
        inputs["spec_json"] = spec_json
    else:
        if not (question_paper and rubric and model_answers):
            raise ValueError("Provide question paper, rubric and model answers (or a JSON spec).")
        inputs["question_paper"] = read_text_document(question_paper)
        inputs["rubric"] = read_text_document(rubric)
        inputs["model_answers"] = read_text_document(model_answers)

    run_id = uuid.uuid4()
    config = {
        "run_name": "auto-assessment",
        "run_id": run_id,
        "tags": ["auto-assessment", settings.provider, *(tags or [])],
        "metadata": {
            "llm": llm.describe(),
            "critic": settings.use_critic,
            "max_grade_retries": settings.max_grade_retries,
            "n_answer_files": len(answer_files),
        },
        "max_concurrency": 4,  # cap parallel graders (rate limits / local GPU)
    }
    return build_graph(llm, settings), inputs, config, run_id


def run_assessment(*, question_paper=None, rubric=None, model_answers=None, spec_file=None,
                   answer_files, settings: Settings | None = None, tags=None) -> dict:
    graph, inputs, config, run_id = _prepare(
        question_paper, rubric, model_answers, spec_file, answer_files, settings, tags)
    state = graph.invoke(inputs, config)
    state["run_id"] = str(run_id)
    return state


def stream_assessment(*, question_paper=None, rubric=None, model_answers=None, spec_file=None,
                      answer_files, settings: Settings | None = None, tags=None) -> Iterator[dict]:
    """Yields {'type': 'progress', 'node': ..., 'update': ...} then {'type': 'final', 'state': ..., 'run_id': ...}."""
    graph, inputs, config, run_id = _prepare(
        question_paper, rubric, model_answers, spec_file, answer_files, settings, tags)
    final: dict = {}
    for mode, chunk in graph.stream(inputs, config, stream_mode=["updates", "values"]):
        if mode == "updates":
            for node, update in chunk.items():
                yield {"type": "progress", "node": node, "update": update}
        else:
            final = chunk
    yield {"type": "final", "state": final, "run_id": str(run_id)}
