"""Public entry points: run the graph (blocking) or stream progress events (for the UI)."""
from __future__ import annotations

import os
import uuid
from typing import Iterator

from .config import Settings
from .graph import build_graph
from .llm_client import create_llm


def tracing_enabled() -> bool:
    flag = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or ""
    return flag.lower() in {"1", "true", "yes"} and bool(
        os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    )


def _prepare(question_paper, rubric, model_answers, model_answers_text, answer_files, settings, tags):
    if not answer_files:
        raise ValueError("Provide at least one answer-sheet file (image / PDF / .txt).")
    if model_answers and model_answers_text:
        raise ValueError("Provide either a model-answer file or model_answers_text, not both.")
    if model_answers_text is not None and not model_answers_text.strip():
        raise ValueError("Model-answer text must not be blank.")
    if not (question_paper and (model_answers or model_answers_text)):
        raise ValueError("Provide question paper and model answers. Rubric is optional.")
    settings = settings or Settings.from_env()
    llm = create_llm(settings)
    inputs: dict = {"answer_files": [str(p) for p in answer_files], "warnings": [], "grades": []}

    inputs["question_paper"] = str(question_paper)
    if rubric:
        inputs["rubric"] = str(rubric)
    if model_answers:
        inputs["model_answers"] = str(model_answers)
    if model_answers_text:
        inputs["model_answers_text"] = model_answers_text

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
        "recursion_limit": max(25, 2 * settings.max_grade_retries + 10),
    }
    return build_graph(llm, settings), inputs, config, run_id


def run_assessment(*, question_paper=None, rubric=None, model_answers=None, model_answers_text=None,
                   answer_files, settings: Settings | None = None, tags=None) -> dict:
    graph, inputs, config, run_id = _prepare(
        question_paper, rubric, model_answers, model_answers_text, answer_files, settings, tags)
    state = graph.invoke(inputs, config)
    state["run_id"] = str(run_id)
    return state


def stream_assessment(*, question_paper=None, rubric=None, model_answers=None, model_answers_text=None,
                      answer_files, settings: Settings | None = None, tags=None) -> Iterator[dict]:
    """Yields {'type': 'progress', 'node': ..., 'update': ...} then {'type': 'final', 'state': ..., 'run_id': ...}."""
    graph, inputs, config, run_id = _prepare(
        question_paper, rubric, model_answers, model_answers_text, answer_files, settings, tags)
    final: dict = {}
    for mode, chunk in graph.stream(inputs, config, stream_mode=["updates", "values"]):
        if mode == "updates":
            for node, update in chunk.items():
                yield {"type": "progress", "node": node, "update": update}
        else:
            final = chunk
    yield {"type": "final", "state": final, "run_id": str(run_id)}
