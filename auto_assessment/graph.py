"""LangGraph workflow assembly and routing."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .llm_client import LLMClient
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
from .models import GraphState, QState


def _route_question_start(state: QState) -> str:
    return "grade" if state["answer"]["attempted"] else "skip"


def _route_after_verification(state: QState, max_grade_retries: int) -> str:
    if state.get("issues") and state["attempts"] <= max_grade_retries:
        return "grade"
    return "finalize"


def build_question_graph(llm: LLMClient, settings: Settings):
    workflow = StateGraph(QState)
    workflow.add_node("grade", GradeQuestionNode(llm, settings))
    workflow.add_node("verify", VerifyQuestionNode(llm, settings))
    workflow.add_node("finalize", FinalizeQuestionNode())
    workflow.add_node("skip", SkipQuestionNode())
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
    question_graph = build_question_graph(llm, settings)
    workflow = StateGraph(GraphState)
    workflow.add_node("ingest", IngestNode(settings))
    workflow.add_node("parse_spec", ParseSpecNode(llm))
    workflow.add_node("clean_align", CleanAlignNode(llm))
    workflow.add_node("grade_question", GradeQuestionsNode(question_graph))
    workflow.add_node("aggregate", AggregateNode())
    workflow.add_node("report", ReportNode(llm))

    workflow.add_edge(START, "ingest")
    workflow.add_edge("ingest", "parse_spec")
    workflow.add_edge("parse_spec", "clean_align")
    workflow.add_edge("clean_align", "grade_question")
    workflow.add_edge("grade_question", "aggregate")
    workflow.add_edge("aggregate", "report")
    workflow.add_edge("report", END)
    return workflow.compile()
