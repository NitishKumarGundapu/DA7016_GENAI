"""Compatibility imports; assessment schemas now live in models."""
from .models import (
    AnswerSegment,
    AssessedCriterion,
    Criterion,
    CriterionScore,
    CriticVerdict,
    ExamSpec,
    GradeDraft,
    OverallFeedback,
    Question,
    QuestionGrade,
    ReportFeedback,
    ReportLabels,
    Segmentation,
)

__all__ = [
    "AnswerSegment", "AssessedCriterion", "Criterion", "CriterionScore",
    "CriticVerdict", "ExamSpec", "GradeDraft", "OverallFeedback", "Question",
    "QuestionGrade", "ReportFeedback", "ReportLabels", "Segmentation",
]
