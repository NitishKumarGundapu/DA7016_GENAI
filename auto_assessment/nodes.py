"""Callable LangGraph nodes for assessment and per-question grading."""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from . import prompts as P
from .config import Settings
from .ingest import extract_document
from .llm_client import LLMClient
from .report import render_report
from .models import (
    Criterion,
    CriticVerdict,
    ExamSpec,
    GradeDraft,
    GraphState,
    Question,
    QState,
    ReportFeedback,
    Segmentation,
)
from .verify import (
    build_grade,
    build_unattempted,
    check_draft,
    evidence_found,
    norm_qid,
    normalize_spec,
)


def _criteria_text(question: Question) -> str:
    return "\n".join(
        f"- {criterion.name} ({criterion.max_marks} marks): {criterion.description}"
        for criterion in question.criteria
    )


class IngestNode:
    def __init__(self, settings: Settings):
        self.settings = settings

    def __call__(self, state: GraphState) -> dict:
        warnings = []

        def extract(paths: list[str], role: str) -> str:
            documents = []
            for index, path in enumerate(paths, start=1):
                text = extract_document(path, self.settings)
                uncertain = text.count("[illegible]") + text.count("[?]")
                if uncertain:
                    warnings.append(
                        f"{role}, document {index}: {uncertain} uncertain words; review source."
                    )
                documents.append(f"=== DOCUMENT {index} ===\n{text}")
            return "\n\n".join(documents)

        output = {}
        for role in ("question_paper", "rubric", "model_answers"):
            if role == "rubric" and not state.get("rubric"):
                continue
            if role == "model_answers" and state.get("model_answers_text"):
                output[role] = state["model_answers_text"]
            else:
                output[role] = extract([state[role]], role)
        output["transcript"] = extract(state["answer_files"], "student answer sheet")
        output["document_count"] = len(state["answer_files"])
        if not output["transcript"].strip():
            raise ValueError("No readable answer-sheet documents found.")
        return {**output, "warnings": warnings}


class ParseSpecNode:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def __call__(self, state: GraphState) -> dict:
        human = P.PARSE_SPEC_HUMAN.format(
            question_paper=state.get("question_paper", ""),
            rubric=state.get("rubric", ""),
            model_answers=state.get("model_answers", ""),
        )
        spec = self.llm.structured(ExamSpec, temperature=0).invoke(
            [SystemMessage(content=P.PARSE_SPEC_SYSTEM), HumanMessage(content=human)]
        )
        if not state.get("rubric"):
            for question in spec.questions:
                question.criteria = [Criterion(
                    name="Whole question",
                    description=(
                        f"Evaluate correctness and completeness against this model answer: {question.model_answer}\n"
                        "Award proportional partial credit for correct content or valid alternative methods; "
                        "full marks require a complete correct answer."
                    ),
                    max_marks=question.max_marks,
                )]
        spec, warnings = normalize_spec(spec)
        return {"spec": spec.model_dump(), "warnings": warnings}


class CleanAlignNode:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def __call__(self, state: GraphState) -> dict:
        spec = ExamSpec.model_validate(state["spec"])
        question_list = "\n".join(f"- id={q.id}: {q.text}" for q in spec.questions)
        segmentation = self.llm.structured(Segmentation, temperature=0).invoke([
            SystemMessage(content=P.SEGMENT_SYSTEM),
            HumanMessage(content=f"QUESTIONS:\n{question_list}\n\nTRANSCRIPT:\n{state['transcript']}"),
        ])
        expected = {norm_qid(question.id) for question in spec.questions}
        by_id = {}
        for answer in segmentation.answers:
            key = norm_qid(answer.question_id)
            if key not in expected or key in by_id:
                raise ValueError(f"Unknown or duplicate segmented question: {answer.question_id}.")
            by_id[key] = answer
        if set(by_id) != expected:
            raise ValueError("Answer alignment omitted questions; cannot treat missing mappings as unanswered.")
        answers, items = [], []
        transcript = re.sub(r"(?m)^=== DOCUMENT \d+ ===\n?", "", state["transcript"])
        for question in spec.questions:
            answer = by_id[norm_qid(question.id)]
            attempted = bool(answer.answer_text.strip())
            if answer.attempted != attempted:
                raise ValueError(f"Q{question.id}: attempted flag contradicts extracted answer text.")
            if attempted and not evidence_found(answer.answer_text, transcript, threshold=100):
                raise ValueError(f"Q{question.id}: aligned answer is not grounded in the student transcript.")
            answers.append({
                "question_id": question.id,
                "attempted": attempted,
                "answer_text": answer.answer_text,
            })
            items.append({
                "question_id": question.id,
                "question": question.text,
                "answer": answer.answer_text,
                "model_answer": question.model_answer,
                "rubric": _criteria_text(question),
                "max_marks": question.max_marks,
            })
        return {"answers": answers, "items": items}


class GradeQuestionsNode:
    def __init__(self, question_graph):
        self.question_graph = question_graph

    def __call__(self, state: GraphState, config: RunnableConfig) -> dict:
        spec = ExamSpec.model_validate(state["spec"])
        grades = []
        for question, answer in zip(spec.questions, state["answers"]):
            output = self.question_graph.invoke(
                {"question": question.model_dump(), "answer": answer}, config
            )
            grades.append(output["grade"])
        return {"grades": grades}


class AggregateNode:
    def __call__(self, state: GraphState) -> dict:
        spec = ExamSpec.model_validate(state["spec"])
        order = {question.id: index for index, question in enumerate(spec.questions)}
        grades = sorted(
            state.get("grades", []),
            key=lambda grade: order.get(grade["question_id"], 10**6),
        )
        total = round(sum(grade["marks_awarded"] for grade in grades), 6)
        max_total = round(sum(question.max_marks for question in spec.questions), 6)
        warnings = [
            f"Q{grade['question_id']}: grading still had open verification issues after "
            f"{grade['attempts']} attempt(s) - see notes."
            for grade in grades if not grade["verified"]
        ]
        result = {
            "title": spec.title,
            "total": total,
            "max_total": max_total,
            "percentage": round(100 * total / max_total, 1) if max_total else 0.0,
            "grades": grades,
        }
        return {"result": result, "warnings": warnings}


class ReportNode:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def __call__(self, state: GraphState) -> dict:
        result = state["result"]
        warnings = state.get("warnings", [])
        diagnostics = [
            *warnings,
            *(note for grade in result["grades"] for note in grade["verification_notes"]),
        ]
        rows = json.dumps([
            {**item, "evaluation": grade}
            for item, grade in zip(state["items"], result["grades"])
        ], ensure_ascii=False)
        overall = self.llm.structured(ReportFeedback, temperature=0.2).invoke([
            SystemMessage(content=P.OVERALL_SYSTEM),
            HumanMessage(content=P.OVERALL_HUMAN.format(
                title=result["title"],
                total=result["total"],
                max_total=result["max_total"],
                rows=rows,
                question_paper="\n".join(item["question"] for item in state["items"]),
                diagnostics=json.dumps(diagnostics, ensure_ascii=False),
            )),
        ])
        translations = overall.diagnostic_translations
        if len(translations) != len(diagnostics):
            raise ValueError("Report omitted diagnostic translations; cannot hide review warnings.")
        grades = []
        offset = len(warnings)
        for original in result["grades"]:
            count = len(original["verification_notes"])
            grade = {
                **original,
                "verification_notes": translations[offset:offset + count],
            }
            offset += count
            if not grade["attempted"]:
                grade["gaps"] = overall.labels.unattempted
                grade["how_to_improve"] = overall.labels.attempt_advice
            grades.append(grade)
        full = {
            **result,
            "grades": grades,
            "warnings": translations[:len(warnings)],
            "overall": overall.model_dump(),
        }
        return {
            "overall": overall.model_dump(),
            "result": full,
            "report_md": render_report(
                full, state.get("transcript", ""), full["warnings"]
            ),
        }


class SkipQuestionNode:
    def __call__(self, state: QState) -> dict:
        question = Question.model_validate(state["question"])
        return {"grade": build_unattempted(question).model_dump()}


class GradeQuestionNode:
    def __init__(self, llm: LLMClient, settings: Settings):
        self.llm = llm
        self.settings = settings

    def __call__(self, state: QState) -> dict:
        question = Question.model_validate(state["question"])
        issues = state.get("issues") or []
        critique = P.RETRY_BLOCK.format(
            issues="\n".join(f"- {issue}" for issue in issues)
        ) if issues else ""
        human = P.GRADE_HUMAN.format(
            item=json.dumps({
                "question_id": question.id,
                "question": question.text,
                "answer": state["answer"]["answer_text"],
                "model_answer": question.model_answer,
                "rubric": _criteria_text(question),
                "max_marks": question.max_marks,
            }, ensure_ascii=False),
            critique=critique,
        )
        draft = self.llm.structured(
            GradeDraft, temperature=self.settings.grader_temperature
        ).invoke([SystemMessage(content=P.GRADE_SYSTEM), HumanMessage(content=human)])
        return {
            "draft": draft.model_dump(),
            "attempts": state.get("attempts", 0) + 1,
        }


class VerifyQuestionNode:
    def __init__(self, llm: LLMClient, settings: Settings):
        self.llm = llm
        self.settings = settings

    def __call__(self, state: QState) -> dict:
        question = Question.model_validate(state["question"])
        answer = state["answer"]["answer_text"]
        draft = GradeDraft.model_validate(state["draft"])
        issues = check_draft(question, answer, draft)
        if "[illegible]" in answer or "[?]" in answer:
            issues.append("Answer contains uncertain transcription; human review required.")
        if not issues and self.settings.use_critic:
            human = P.CRITIC_HUMAN.format(
                qid=question.id,
                max_marks=question.max_marks,
                question=question.text,
                criteria=_criteria_text(question),
                model_answer=question.model_answer,
                answer=answer,
                grading=draft.model_dump_json(indent=1),
            )
            verdict = self.llm.structured(CriticVerdict, temperature=0).invoke(
                [SystemMessage(content=P.CRITIC_SYSTEM), HumanMessage(content=human)]
            )
            if not verdict.ok:
                issues = list(verdict.issues) or ["Critic rejected the grading."]
        return {"issues": issues}


class FinalizeQuestionNode:
    def __call__(self, state: QState) -> dict:
        question = Question.model_validate(state["question"])
        grade = build_grade(
            question,
            state["answer"]["answer_text"],
            GradeDraft.model_validate(state["draft"]),
            state["attempts"],
            state.get("issues") or [],
        )
        return {"grade": grade.model_dump()}
