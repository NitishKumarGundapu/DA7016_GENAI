"""Pydantic models used for LLM structured output and internal results.

The ``Field(description=...)`` strings are sent to the model and are part of the
prompt contract: keep them precise when changing the grading rules.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- exam spec
class Criterion(BaseModel):
    """One rubric criterion and the marks explicitly allocated to it."""

    name: str = Field(description="Short unique label, e.g. 'Defines gradient descent'")
    description: str = Field(description="What the student must show to earn these marks")
    max_marks: float


class Question(BaseModel):
    """One question aligned across the question paper, model answers, and rubric."""

    id: str = Field(description="Question label exactly as printed: '1', '2a', 'Q3(b)' ...")
    text: str = Field(description="Full question text")
    max_marks: float
    model_answer: str = Field(description="Reference / model answer for this question")
    criteria: list[Criterion] = Field(
        description="Rubric criteria. Their max_marks must sum to the question's max_marks. "
        "Only use explicitly supplied criterion marks. Never infer or rescale allocations."
    )


class ExamSpec(BaseModel):
    """The complete question paper, in printed order."""

    title: str
    questions: list[Question]


# ------------------------------------------------------------------ transcription/alignment
class AnswerSegment(BaseModel):
    """A student's answer to exactly one question, copied from the transcript."""

    question_id: str = Field(description="Exactly one of the provided question ids")
    attempted: bool
    answer_text: str = Field(description="The student's answer copied from the transcript, verbatim")


class Segmentation(BaseModel):
    """All answer segments returned by the alignment step."""

    answers: list[AnswerSegment]


# --------------------------------------------------------------------------- grading
class CriterionScore(BaseModel):
    """A mark proposed by the LLM for one rubric criterion, with evidence."""

    criterion: str = Field(description="Criterion name exactly as given in the rubric")
    marks_awarded: float
    evidence: str = Field(
        description="Short VERBATIM quote (<=25 words) from the student's answer that earns the "
        "marks. Empty string if 0 marks."
    )
    justification: str = Field(description="One or two sentences tying the evidence to the criterion")


class GradeDraft(BaseModel):
    """The first-pass grading of one question, before deterministic verification."""

    criterion_scores: list[CriterionScore]
    strengths: str = Field(description="What this student did well, quoting their own answer")
    gaps: str = Field(
        description="Exactly what was missing/wrong, naming the rubric criterion and marks lost"
    )
    how_to_improve: str = Field(
        description="Concrete corrections (state the right idea briefly) and one practice suggestion"
    )


class CriticVerdict(BaseModel):
    """Second-opinion review of a grade draft."""

    ok: bool
    issues: list[str] = Field(description="Short, actionable problems; empty when ok")


class OverallFeedback(BaseModel):
    """Student-facing summary of the whole paper."""

    summary: str = Field(description="3-4 sentence overall assessment of the whole paper")
    top_priorities: list[str] = Field(description="At most 3 study priorities, each citing question ids")


class ReportLabels(BaseModel):
    """Report vocabulary translated by the report node, never student scores."""

    assessment_report: str
    total: str
    study_priorities: str
    marks_by_question: str
    question: str
    marks: str
    maximum: str
    withheld: str
    attempted: str
    verified: str
    detailed_feedback: str
    criterion: str
    evidence: str
    justification: str
    strengths: str
    gaps: str
    improvement: str
    verification_notes: str
    warnings: str
    transcript: str
    yes: str
    no: str
    unattempted: str
    attempt_advice: str


class ReportFeedback(OverallFeedback):
    """Overall feedback plus localized report labels and diagnostics."""

    labels: ReportLabels
    diagnostic_translations: list[str] = Field(
        description="Every supplied diagnostic translated into report language, in original order. "
        "Preserve question ids, counts, uncertainty and review requirements. Never omit any."
    )


# --------------------------------------------------------------------------- internal results
class AssessedCriterion(CriterionScore):
    """A criterion score after clamping, grounding checks, and withheld-mark calculation."""

    max_marks: float
    marks_withheld: float


class QuestionGrade(BaseModel):
    """Final, code-validated result for one question."""

    question_id: str
    max_marks: float
    marks_awarded: float
    attempted: bool
    criterion_scores: list[AssessedCriterion] = Field(default_factory=list)
    strengths: str = ""
    gaps: str = ""
    how_to_improve: str = ""
    answer_text: str = ""
    attempts: int = 0
    verified: bool = True
    verification_notes: list[str] = Field(default_factory=list)


class GraphState(TypedDict, total=False):
    """Top-level graph state.

    ``warnings`` is additive: LangGraph merges lists returned by parallel/successive
    nodes instead of replacing the previous value.
    """

    question_paper: str
    rubric: str
    model_answers: str
    model_answers_text: str
    answer_files: list[str]

    spec: dict
    transcript: str
    document_count: int
    answers: list[dict]
    items: list[dict]
    grades: list[dict]
    warnings: Annotated[list[str], operator.add]

    result: dict
    overall: dict
    report_md: str


class QState(TypedDict, total=False):
    """State for the per-question grade -> verify -> finalize sub-graph."""

    question: dict
    answer: dict
    draft: dict
    issues: list[str]
    attempts: int
    grade: dict
