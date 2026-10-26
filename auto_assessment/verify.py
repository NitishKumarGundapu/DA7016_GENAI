"""Deterministic checks. The LLM proposes marks; code enforces arithmetic and grounding."""
from __future__ import annotations

import re

from .schemas import ExamSpec, GradeDraft, Question, QuestionGrade


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def norm_qid(s: str) -> str:
    """'Q2(a)', '2 a', '2.a' -> '2a'"""
    s = s.lower().strip()
    s = re.sub(r"^(question|q)\s*", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def evidence_found(evidence: str, answer: str, threshold: int = 85) -> bool:
    e, a = _norm(evidence), _norm(answer)
    if not e:
        return False
    if e in a:
        return True
    try:
        from rapidfuzz import fuzz

        return fuzz.partial_ratio(e, a) >= threshold
    except ImportError:  # pragma: no cover
        from difflib import SequenceMatcher

        return SequenceMatcher(None, e, a).find_longest_match(0, len(e), 0, len(a)).size >= 0.8 * len(e)


# ------------------------------------------------------------------ spec normalisation
def normalize_spec(spec: ExamSpec) -> tuple[ExamSpec, list[str]]:
    """Make criteria marks sum exactly to each question's max_marks; return warnings."""
    warnings: list[str] = []
    for q in spec.questions:
        total = sum(c.max_marks for c in q.criteria)
        if not q.criteria:
            warnings.append(f"Q{q.id}: no rubric criteria found.")
        elif abs(total - q.max_marks) > 1e-6:
            warnings.append(f"Q{q.id}: criteria sum {total} != max {q.max_marks}; rescaled.")
            factor = q.max_marks / total if total else 0
            for c in q.criteria:
                c.max_marks = round(c.max_marks * factor, 2)
    return spec, warnings


# ------------------------------------------------------------------ grade checks
def check_draft(q: Question, answer: str, draft: GradeDraft) -> list[str]:
    issues: list[str] = []
    rubric = {c.name.strip().lower(): c for c in q.criteria}
    seen: set[str] = set()
    for cs in draft.criterion_scores:
        key = cs.criterion.strip().lower()
        c = rubric.get(key)
        if c is None:
            issues.append(f"Unknown criterion '{cs.criterion}'. Use rubric names exactly.")
            continue
        if key in seen:
            issues.append(f"Criterion '{cs.criterion}' scored more than once.")
        seen.add(key)
        if cs.marks_awarded < 0 or cs.marks_awarded > c.max_marks + 1e-6:
            issues.append(f"'{c.name}': {cs.marks_awarded} outside 0..{c.max_marks}.")
        if cs.marks_awarded > 0 and not evidence_found(cs.evidence, answer):
            issues.append(f"'{c.name}': marks given but evidence quote not found verbatim in the answer.")
    for key, c in rubric.items():
        if key not in seen:
            issues.append(f"Criterion '{c.name}' was not scored.")
    if not draft.gaps.strip() and sum(s.marks_awarded for s in draft.criterion_scores) < q.max_marks:
        issues.append("Marks were lost but 'gaps' does not explain where.")
    return issues


def build_grade(q: Question, answer: str, draft: GradeDraft, attempts: int, issues: list[str]) -> QuestionGrade:
    """Clamp to rubric bounds and compute the total in code (never trust LLM arithmetic)."""
    rubric = {c.name.strip().lower(): c for c in q.criteria}
    scores = []
    for cs in draft.criterion_scores:
        c = rubric.get(cs.criterion.strip().lower())
        if c is None:
            continue
        cs = cs.model_copy(update={"marks_awarded": round(min(max(cs.marks_awarded, 0), c.max_marks), 2)})
        scores.append(cs)
    total = round(min(sum(s.marks_awarded for s in scores), q.max_marks), 2)
    return QuestionGrade(
        question_id=q.id, max_marks=q.max_marks, marks_awarded=total, attempted=True,
        criterion_scores=scores, strengths=draft.strengths, gaps=draft.gaps,
        how_to_improve=draft.how_to_improve, answer_text=answer, attempts=attempts,
        verified=not issues, verification_notes=issues,
    )


def build_unattempted(q: Question) -> QuestionGrade:
    needed = "; ".join(f"{c.name} ({c.max_marks})" for c in q.criteria)
    return QuestionGrade(
        question_id=q.id, max_marks=q.max_marks, marks_awarded=0.0, attempted=False,
        gaps="No answer was found for this question, so no marks could be awarded.",
        how_to_improve=f"A full-mark answer needs: {needed}. Attempt every question, even partially - "
                       "method marks are available.",
    )
