"""Pydantic schemas. LLM-facing models have no defaults (keeps tool schemas strict-friendly)."""
from __future__ import annotations

from pydantic import BaseModel, Field


# ------------------------------------------------------------- exam specification
class Criterion(BaseModel):
    name: str = Field(description="Short unique label, e.g. 'Defines gradient descent'")
    description: str = Field(description="What the student must show to earn these marks")
    max_marks: float


class Question(BaseModel):
    id: str = Field(description="Question label exactly as printed: '1', '2a', 'Q3(b)' ...")
    text: str = Field(description="Full question text")
    max_marks: float
    model_answer: str = Field(description="Reference / model answer for this question")
    criteria: list[Criterion] = Field(
        description="Rubric criteria. Their max_marks must sum to the question's max_marks. "
        "If the rubric gives no breakdown, split the marks into sensible criteria."
    )


class ExamSpec(BaseModel):
    title: str
    questions: list[Question]


# ------------------------------------------------------------- transcription / segmentation
class PageTranscript(BaseModel):
    transcript: str = Field(description="Faithful transcription of the page, line by line")
    illegible_count: int = Field(description="Number of words marked [illegible] or [?]")


class AnswerSegment(BaseModel):
    question_id: str = Field(description="Exactly one of the provided question ids")
    attempted: bool
    answer_text: str = Field(description="The student's answer copied from the transcript, verbatim")


class Segmentation(BaseModel):
    answers: list[AnswerSegment]


# ------------------------------------------------------------- grading
class CriterionScore(BaseModel):
    criterion: str = Field(description="Criterion name exactly as given in the rubric")
    marks_awarded: float
    evidence: str = Field(
        description="Short VERBATIM quote (<=25 words) from the student's answer that earns the "
        "marks. Empty string if 0 marks."
    )
    justification: str = Field(description="One or two sentences tying the evidence to the criterion")


class GradeDraft(BaseModel):
    criterion_scores: list[CriterionScore]
    strengths: str = Field(description="What this student did well, quoting their own answer")
    gaps: str = Field(
        description="Exactly what was missing/wrong, naming the rubric criterion and marks lost"
    )
    how_to_improve: str = Field(
        description="Concrete corrections (state the right idea briefly) and one practice suggestion"
    )


class CriticVerdict(BaseModel):
    ok: bool
    issues: list[str] = Field(description="Short, actionable problems; empty when ok")


class OverallFeedback(BaseModel):
    summary: str = Field(description="3-4 sentence overall assessment of the whole paper")
    top_priorities: list[str] = Field(description="At most 3 study priorities, each citing question ids")


# ------------------------------------------------------------- internal result models
class QuestionGrade(BaseModel):
    question_id: str
    max_marks: float
    marks_awarded: float
    attempted: bool
    criterion_scores: list[CriterionScore] = []
    strengths: str = ""
    gaps: str = ""
    how_to_improve: str = ""
    answer_text: str = ""
    attempts: int = 0
    verified: bool = True
    verification_notes: list[str] = []
