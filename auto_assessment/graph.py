"""The LangGraph workflow.

                 +--> parse_spec --+
        START ---+                 +--> segment --(Send x N questions)--> grade_question --> aggregate --> summarize --> END
                 +--> transcribe --+                                       (sub-graph: grade -> verify -> retry?)

* parse_spec and transcribe run in parallel (no dependency between them).
* grade_question fans out with Send: one parallel sub-graph run per question.
* Each sub-graph is a generate -> verify -> (retry with critique) loop = the "agentic" part.
"""
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import prompts as P
from .config import Settings
from .ingest import load_page_images, load_typed_text
from .llm import HybridLLM
from .report import render_report
from .schemas import (AnswerSegment, CriticVerdict, ExamSpec, GradeDraft, OverallFeedback,
                      PageTranscript, Question, Segmentation)
from .verify import (build_grade, build_unattempted, check_draft, evidence_found, norm_qid,
                     normalize_spec)


# ======================================================================= state
class GraphState(TypedDict, total=False):
    # inputs
    question_paper: str
    rubric: str
    model_answers: str
    spec_json: dict          # optional pre-structured spec (skips the LLM parse)
    answer_files: list[str]
    # intermediate
    spec: dict
    transcript: str
    page_count: int
    answers: list[dict]
    grades: Annotated[list[dict], operator.add]     # fan-in from parallel graders
    warnings: Annotated[list[str], operator.add]
    # outputs
    result: dict
    overall: dict
    report_md: str


class QState(TypedDict, total=False):
    question: dict
    answer: dict
    draft: dict
    issues: list[str]
    attempts: int
    grade: dict


def _criteria_text(q: Question) -> str:
    return "\n".join(f"- {c.name} ({c.max_marks} marks): {c.description}" for c in q.criteria)


# ======================================================================= per-question sub-graph
def build_question_graph(llm: HybridLLM, s: Settings):
    def route_start(qs: QState) -> str:
        return "grade" if qs["answer"]["attempted"] else "skip"

    def skip(qs: QState):
        q = Question.model_validate(qs["question"])
        return {"grade": build_unattempted(q).model_dump()}

    def grade(qs: QState):
        q = Question.model_validate(qs["question"])
        issues = qs.get("issues") or []
        critique = P.RETRY_BLOCK.format(issues="\n".join(f"- {i}" for i in issues)) if issues else ""
        human = P.GRADE_HUMAN.format(
            qid=q.id, max_marks=q.max_marks, question=q.text, criteria=_criteria_text(q),
            model_answer=q.model_answer, answer=qs["answer"]["answer_text"], critique=critique,
        )
        draft = llm.structured(GradeDraft, temperature=s.grader_temperature).invoke(
            [SystemMessage(content=P.GRADE_SYSTEM), HumanMessage(content=human)]
        )
        return {"draft": draft.model_dump(), "attempts": qs.get("attempts", 0) + 1}

    def verify(qs: QState):
        q = Question.model_validate(qs["question"])
        answer = qs["answer"]["answer_text"]
        draft = GradeDraft.model_validate(qs["draft"])
        issues = check_draft(q, answer, draft)                    # 1) deterministic checks
        if not issues and s.use_critic:                           # 2) LLM critic
            human = P.CRITIC_HUMAN.format(
                qid=q.id, max_marks=q.max_marks, question=q.text, criteria=_criteria_text(q),
                model_answer=q.model_answer, answer=answer, grading=draft.model_dump_json(indent=1),
            )
            verdict = llm.structured(CriticVerdict, temperature=0).invoke(
                [SystemMessage(content=P.CRITIC_SYSTEM), HumanMessage(content=human)]
            )
            if not verdict.ok:
                issues = list(verdict.issues) or ["Critic rejected the grading."]
        return {"issues": issues}

    def after_verify(qs: QState) -> str:
        return "grade" if qs.get("issues") and qs["attempts"] <= s.max_grade_retries else "finalize"

    def finalize(qs: QState):
        q = Question.model_validate(qs["question"])
        g = build_grade(q, qs["answer"]["answer_text"], GradeDraft.model_validate(qs["draft"]),
                        qs["attempts"], qs.get("issues") or [])
        return {"grade": g.model_dump()}

    g = StateGraph(QState)
    g.add_node("grade", grade)
    g.add_node("verify", verify)
    g.add_node("finalize", finalize)
    g.add_node("skip", skip)
    g.add_conditional_edges(START, route_start, {"grade": "grade", "skip": "skip"})
    g.add_edge("grade", "verify")
    g.add_conditional_edges("verify", after_verify, {"grade": "grade", "finalize": "finalize"})
    g.add_edge("finalize", END)
    g.add_edge("skip", END)
    return g.compile()


# ======================================================================= main graph
def build_graph(llm: HybridLLM, s: Settings):
    q_graph = build_question_graph(llm, s)

    def parse_spec(state: GraphState):
        if state.get("spec_json"):
            spec = ExamSpec.model_validate(state["spec_json"])
        else:
            human = P.PARSE_SPEC_HUMAN.format(
                question_paper=state.get("question_paper", ""),
                rubric=state.get("rubric", ""),
                model_answers=state.get("model_answers", ""),
            )
            spec = llm.structured(ExamSpec, temperature=0).invoke(
                [SystemMessage(content=P.PARSE_SPEC_SYSTEM), HumanMessage(content=human)]
            )
        spec, warns = normalize_spec(spec)
        return {"spec": spec.model_dump(), "warnings": warns}

    def transcribe(state: GraphState):
        files = state["answer_files"]
        pages = load_page_images(files, dpi=s.pdf_dpi, max_side=s.max_image_side)
        if not pages:  # typed .txt/.md answer sheet (handy for testing without scans)
            text = load_typed_text(files)
            if not text.strip():
                raise ValueError("No readable answer-sheet pages found.")
            return {"transcript": text, "page_count": 1}
        chain = llm.structured(PageTranscript, vision=True, temperature=0)
        batch = [
            [SystemMessage(content=P.TRANSCRIBE_SYSTEM),
             HumanMessage(content=[
                 {"type": "text", "text": f"Transcribe page {p['page']} ({p['source']})."},
                 {"type": "image_url", "image_url": {"url": p["data_url"]}},
             ])]
            for p in pages
        ]
        results = chain.batch(batch, config={"max_concurrency": s.transcribe_concurrency})
        parts = [f"=== PAGE {p['page']} ===\n{r.transcript.strip()}" for p, r in zip(pages, results)]
        warns = [f"Page {p['page']}: {r.illegible_count} illegible/uncertain words in transcription."
                 for p, r in zip(pages, results) if r.illegible_count > 0]
        return {"transcript": "\n\n".join(parts), "page_count": len(pages), "warnings": warns}

    def segment(state: GraphState):
        spec = ExamSpec.model_validate(state["spec"])
        qlist = "\n".join(f"- id={q.id}: {q.text}" for q in spec.questions)
        seg = llm.structured(Segmentation, temperature=0).invoke([
            SystemMessage(content=P.SEGMENT_SYSTEM),
            HumanMessage(content=f"QUESTIONS:\n{qlist}\n\nTRANSCRIPT:\n{state['transcript']}"),
        ])
        by_id = {norm_qid(a.question_id): a for a in seg.answers}
        answers, warns = [], []
        for q in spec.questions:
            a = by_id.get(norm_qid(q.id)) or AnswerSegment(question_id=q.id, attempted=False, answer_text="")
            attempted = bool(a.attempted and a.answer_text.strip())
            if attempted and not evidence_found(a.answer_text[:200], state["transcript"], threshold=80):
                warns.append(f"Q{q.id}: segmented answer does not closely match the transcript.")
            answers.append({"question_id": q.id, "attempted": attempted, "answer_text": a.answer_text})
        return {"answers": answers, "warnings": warns}

    def fan_out(state: GraphState):
        spec = ExamSpec.model_validate(state["spec"])
        ans = {a["question_id"]: a for a in state["answers"]}
        sends = [Send("grade_question", {"question": q.model_dump(), "answer": ans[q.id]})
                 for q in spec.questions]
        return sends or "aggregate"

    def grade_question(payload: dict, config: RunnableConfig):
        out = q_graph.invoke(payload, config)
        return {"grades": [out["grade"]]}

    def aggregate(state: GraphState):
        spec = ExamSpec.model_validate(state["spec"])
        order = {q.id: i for i, q in enumerate(spec.questions)}
        grades = sorted(state.get("grades", []), key=lambda g: order.get(g["question_id"], 10**6))
        total = round(sum(g["marks_awarded"] for g in grades), 2)       # arithmetic done in code
        max_total = round(sum(q.max_marks for q in spec.questions), 2)
        warns = [f"Q{g['question_id']}: grading still had open verification issues after "
                 f"{g['attempts']} attempt(s) - see notes." for g in grades if not g["verified"]]
        result = {"title": spec.title, "total": total, "max_total": max_total,
                  "percentage": round(100 * total / max_total, 1) if max_total else 0.0,
                  "grades": grades}
        return {"result": result, "warnings": warns}

    def summarize(state: GraphState):
        res = state["result"]
        rows = "\n".join(
            f"Q{g['question_id']}: {g['marks_awarded']}/{g['max_marks']} | gaps: {g['gaps'] or '-'}"
            for g in res["grades"]
        )
        overall = llm.structured(OverallFeedback, temperature=0.2).invoke([
            SystemMessage(content=P.OVERALL_SYSTEM),
            HumanMessage(content=P.OVERALL_HUMAN.format(
                title=res["title"], total=res["total"], max_total=res["max_total"], rows=rows)),
        ])
        full = {**res, "overall": overall.model_dump()}
        return {"overall": overall.model_dump(), "result": full,
                "report_md": render_report(full, state.get("transcript", ""), state.get("warnings", []))}

    g = StateGraph(GraphState)
    g.add_node("parse_spec", parse_spec)
    g.add_node("transcribe", transcribe)
    g.add_node("segment", segment)
    g.add_node("grade_question", grade_question)
    g.add_node("aggregate", aggregate)
    g.add_node("summarize", summarize)

    g.add_edge(START, "parse_spec")
    g.add_edge(START, "transcribe")
    g.add_edge(["parse_spec", "transcribe"], "segment")       # join: wait for both
    g.add_conditional_edges("segment", fan_out, ["grade_question", "aggregate"])
    g.add_edge("grade_question", "aggregate")
    g.add_edge("aggregate", "summarize")
    g.add_edge("summarize", END)
    return g.compile()
