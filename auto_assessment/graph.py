"""LangGraph workflow assembly and routing.

Two graphs are built here:

* :func:`build_graph` is the top-level assessment pipeline.
* :func:`build_question_graph` is the per-question grade/verify/retry loop that the
  top-level graph invokes for every question.
"""
from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .llm_client import LLMClient
from .models import GraphState, QState
from .nodes import (
    AggregateNode,
    CleanAlignNode,
    FinalizeQuestionNode,
    GradeQuestionNode,
    GradeQuestionsNode,
    IngestNode,
    ParseSpecNode,
    ReportNode,
    SkipQuestionNode,
    VerifyQuestionNode,
)


logger = logging.getLogger(__name__)


def _configure_node_logging() -> None:
    """Install a concise node logger unless the application already configured logging."""
    logger.setLevel(logging.INFO)
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)


def _logged_node(name: str, node: Callable, *, accepts_config: bool = False) -> Callable:
    """Wrap a node with start/finish/failure logs, including per-question context."""
    def invoke(state, config=None):
        """Run one node and log its outcome."""
        question = state.get("question") or {}
        answer = state.get("answer") or {}
        question_id = question.get("id") or answer.get("question_id")
        context = f" question={question_id}" if question_id is not None else ""
        if name == "grade":
            context += f" attempt={state.get('attempts', 0) + 1}"

        started = perf_counter()
        logger.info("Starting node=%s%s", name, context)
        try:
            result = node(state, config) if accepts_config else node(state)
        except Exception:
            logger.exception(
                "Node failed node=%s%s elapsed_ms=%.1f",
                name,
                context,
                (perf_counter() - started) * 1000,
            )
            raise

        fields = ",".join(result) if isinstance(result, dict) else "none"
        logger.info(
            "Finished node=%s%s elapsed_ms=%.1f output_fields=%s",
            name,
            context,
            (perf_counter() - started) * 1000,
            fields,
        )
        return result

    return invoke


def _route_question_start(state: QState) -> str:
    """Skip the LLM entirely when alignment found no attempt."""
    return "grade" if state["answer"]["attempted"] else "skip"


def _route_after_verification(state: QState, max_grade_retries: int) -> str:
    """Retry grading while verification issues remain and retries are available."""
    if state.get("issues") and state["attempts"] <= max_grade_retries:
        return "grade"
    return "finalize"


def build_question_graph(llm: LLMClient, settings: Settings):
    """Compile the per-question grade -> verify -> retry/finalize workflow."""
    _configure_node_logging()
    workflow = StateGraph(QState)

    workflow.add_node("grade", _logged_node("grade", GradeQuestionNode(llm, settings)))
    workflow.add_node("verify", _logged_node("verify", VerifyQuestionNode(llm, settings)))
    workflow.add_node("finalize", _logged_node("finalize", FinalizeQuestionNode()))
    workflow.add_node("skip", _logged_node("skip", SkipQuestionNode()))

    workflow.add_conditional_edges(
        START, _route_question_start, {"grade": "grade", "skip": "skip"}
    )
    workflow.add_edge("grade", "verify")
    workflow.add_conditional_edges(
        "verify",
        lambda state: _route_after_verification(state, settings.max_grade_retries),
        {"grade": "grade", "finalize": "finalize"},
    )
    workflow.add_edge("finalize", END)
    workflow.add_edge("skip", END)
    return workflow.compile()


def build_graph(llm: LLMClient, settings: Settings):
    """Compile the complete assessment pipeline described in the README."""
    _configure_node_logging()
    question_graph = build_question_graph(llm, settings)
    workflow = StateGraph(GraphState)

    workflow.add_node("ingest", _logged_node("ingest", IngestNode(settings)))
    workflow.add_node("parse_spec", _logged_node("parse_spec", ParseSpecNode(llm, settings)))
    workflow.add_node("clean_align", _logged_node("clean_align", CleanAlignNode(llm, settings)))
    workflow.add_node(
        "grade_question",
        _logged_node(
            "grade_question",
            GradeQuestionsNode(question_graph),
            accepts_config=True,
        ),
    )
    workflow.add_node("aggregate", _logged_node("aggregate", AggregateNode()))
    workflow.add_node("report", _logged_node("report", ReportNode(llm)))

    workflow.add_edge(START, "ingest")
    workflow.add_edge("ingest", "parse_spec")
    workflow.add_edge("parse_spec", "clean_align")
    workflow.add_edge("clean_align", "grade_question")
    workflow.add_edge("grade_question", "aggregate")
    workflow.add_edge("aggregate", "report")
    workflow.add_edge("report", END)
    return workflow.compile()
