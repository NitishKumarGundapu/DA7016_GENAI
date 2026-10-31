"""Deterministic checks that constrain what the LLM is allowed to award.

The model proposes criterion marks; this module enforces arithmetic bounds,
verbatim evidence, and complete criterion coverage. Unresolved problems become
review notes instead of silently accepted marks.
"""
from __future__ import annotations

import math
import re
import unicodedata

from .models import (
    AssessedCriterion,
    CriterionScore,
    ExamSpec,
    GradeDraft,
    Question,
    QuestionGrade,
)

# Minimum rapidfuzz partial-ratio for evidence that is not an exact substring.
EVIDENCE_MATCH_THRESHOLD = 85


def _norm(text: str) -> str:
    """Casefold and strip punctuation/formatting so quotes can be compared fairly."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    return " ".join(
        "".join(
            char if unicodedata.category(char)[0] in "LNM" else " " for char in normalized
        ).split()
    )


def norm_qid(value: str) -> str:
    """Normalize question ids: ``'Q2(a)'``, ``'2 a'`` and ``'2.a'`` all become ``'2a'``."""
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = re.sub(r"^(question|q)\s*", "", value)
    return "".join(
        char for char in value
        if char.isalnum() or unicodedata.category(char).startswith("M")
    )


def evidence_found(evidence: str, answer: str, threshold: int = EVIDENCE_MATCH_THRESHOLD) -> bool:
    """Return True when ``evidence`` appears in ``answer`` (exactly or fuzzily)."""
    normalized_evidence, normalized_answer = _norm(evidence), _norm(answer)
    if not normalized_evidence or not normalized_answer:
        return False
    if normalized_evidence in normalized_answer:
        return True
    try:
        from rapidfuzz import fuzz

        return fuzz.partial_ratio(normalized_evidence, normalized_answer) >= threshold
    except ImportError:  # pragma: no cover - rapidfuzz is a declared dependency
        from difflib import SequenceMatcher

        match = SequenceMatcher(
            None, normalized_evidence, normalized_answer
        ).find_longest_match(0, len(normalized_evidence), 0, len(normalized_answer))
        return match.size >= threshold / 100 * len(normalized_evidence)


# --------------------------------------------------------------------------- spec validation
def normalize_spec(spec: ExamSpec) -> tuple[ExamSpec, list[str]]:
    """Validate supplied marks without inventing or rescaling rubric allocations.

    Raises ``ValueError`` for unusable papers; ``warnings`` is currently always empty
    and kept for a stable return contract.
    """
    if not spec.questions:
        raise ValueError("No questions found in the question paper/spec.")

    seen: set[str] = set()
    for question in spec.questions:
        key = norm_qid(question.id)
        if not key or key in seen:
            raise ValueError(f"Question id is empty or duplicated: {question.id!r}.")
        seen.add(key)

        if not question.text.strip() or not question.model_answer.strip():
            raise ValueError(f"Q{question.id}: question and model answer are required.")
        if not math.isfinite(question.max_marks) or question.max_marks <= 0:
            raise ValueError(f"Q{question.id}: maximum marks must be finite and positive.")

        names = [criterion.name.strip().casefold() for criterion in question.criteria]
        if not question.criteria or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError(f"Q{question.id}: explicit, uniquely named rubric criteria are required.")

        for criterion in question.criteria:
            if (
                not criterion.description.strip()
                or not math.isfinite(criterion.max_marks)
                or criterion.max_marks <= 0
            ):
                raise ValueError(
                    f"Q{question.id}, {criterion.name}: explicit positive criterion marks "
                    "and description required."
                )

        total = sum(criterion.max_marks for criterion in question.criteria)
        if abs(total - question.max_marks) > 1e-6:
            raise ValueError(
                f"Q{question.id}: criterion marks sum to {total}, not {question.max_marks}. "
                "Fix the rubric."
            )

    return spec, []


# --------------------------------------------------------------------------- grade checks
def check_draft(question: Question, answer: str, draft: GradeDraft) -> list[str]:
    """Return every problem that prevents accepting a grade draft as-is."""
    issues: list[str] = []
    rubric = {criterion.name.strip().lower(): criterion for criterion in question.criteria}
    seen: set[str] = set()

    for criterion_score in draft.criterion_scores:
        key = criterion_score.criterion.strip().lower()
        criterion = rubric.get(key)
        if criterion is None:
            issues.append(
                f"Unknown criterion '{criterion_score.criterion}'. Use rubric names exactly."
            )
            continue
        if key in seen:
            issues.append(f"Criterion '{criterion_score.criterion}' scored more than once.")
        seen.add(key)

        marks = criterion_score.marks_awarded
        if not math.isfinite(marks) or marks < 0 or marks > criterion.max_marks + 1e-6:
            issues.append(f"'{criterion.name}': {marks} outside 0..{criterion.max_marks}.")
        if marks > 0 and not evidence_found(criterion_score.evidence, answer):
            issues.append(
                f"'{criterion.name}': marks given but evidence quote not found verbatim "
                "in the answer."
            )

    for key, criterion in rubric.items():
        if key not in seen:
            issues.append(f"Criterion '{criterion.name}' was not scored.")

    if not draft.gaps.strip() and sum(
        score.marks_awarded for score in draft.criterion_scores
    ) < question.max_marks:
        issues.append("Marks were lost but 'gaps' does not explain where.")

    return issues


def build_grade(
    question: Question,
    answer: str,
    draft: GradeDraft,
    attempts: int,
    issues: list[str],
) -> QuestionGrade:
    """Compute one bounded score per criterion, keeping unresolved issues for review."""
    rubric = {criterion.name.strip().lower(): criterion for criterion in question.criteria}
    by_name: dict[str, CriterionScore] = {}
    for criterion_score in draft.criterion_scores:
        key = criterion_score.criterion.strip().lower()
        if key in rubric and key not in by_name:
            by_name[key] = criterion_score

    scores: list[AssessedCriterion] = []
    for key, criterion in rubric.items():
        criterion_score = by_name.get(key)
        marks = (
            min(max(criterion_score.marks_awarded, 0), criterion.max_marks)
            if criterion_score and math.isfinite(criterion_score.marks_awarded)
            else 0.0
        )
        if criterion_score and marks > 0 and not evidence_found(criterion_score.evidence, answer):
            marks = 0.0
        scores.append(
            AssessedCriterion(
                criterion=criterion.name,
                marks_awarded=marks,
                max_marks=criterion.max_marks,
                marks_withheld=round(criterion.max_marks - marks, 6),
                evidence=criterion_score.evidence if criterion_score and marks > 0 else "",
                justification=criterion_score.justification if criterion_score else criterion.description,
            )
        )

    total = round(sum(score.marks_awarded for score in scores), 6)
    return QuestionGrade(
        question_id=question.id,
        max_marks=question.max_marks,
        marks_awarded=total,
        attempted=True,
        criterion_scores=scores,
        strengths=draft.strengths,
        gaps=draft.gaps,
        how_to_improve=draft.how_to_improve,
        answer_text=answer,
        attempts=attempts,
        verified=not issues,
        verification_notes=issues,
    )


def build_unattempted(question: Question) -> QuestionGrade:
    """Return a zero-mark grade for a question the student did not attempt."""
    return QuestionGrade(
        question_id=question.id,
        max_marks=question.max_marks,
        marks_awarded=0.0,
        attempted=False,
        criterion_scores=[
            AssessedCriterion(
                criterion=criterion.name,
                marks_awarded=0,
                max_marks=criterion.max_marks,
                marks_withheld=criterion.max_marks,
                evidence="",
                justification=criterion.description,
            )
            for criterion in question.criteria
        ],
    )


def build_failed(question: Question, note: str, attempted: bool = True) -> QuestionGrade:
    """Placeholder for a question whose grading failed; marks are withheld for review."""
    return QuestionGrade(
        question_id=question.id,
        max_marks=question.max_marks,
        marks_awarded=0.0,
        attempted=attempted,
        verified=False,
        verification_notes=[note],
    )
