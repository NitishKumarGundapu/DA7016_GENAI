"""Deterministic checks. The LLM proposes marks; code enforces arithmetic and grounding."""
from __future__ import annotations

import re
import math
import unicodedata

from .models import AssessedCriterion, ExamSpec, GradeDraft, Question, QuestionGrade


def _norm(s: str) -> str:
    text = unicodedata.normalize("NFC", s).casefold()
    return " ".join("".join(
        ch if unicodedata.category(ch)[0] in "LNM" else " " for ch in text
    ).split())


def norm_qid(s: str) -> str:
    """'Q2(a)', '2 a', '2.a' -> '2a'"""
    s = unicodedata.normalize("NFKC", s).casefold().strip()
    s = re.sub(r"^(question|q)\s*", "", s)
    return "".join(ch for ch in s if ch.isalnum() or unicodedata.category(ch).startswith("M"))


def evidence_found(evidence: str, answer: str, threshold: int = 85) -> bool:
    e, a = _norm(evidence), _norm(answer)
    if not e or not a:
        return False
    if e in a:
        return True
    try:
        from rapidfuzz import fuzz

        return fuzz.partial_ratio(e, a) >= threshold
    except ImportError:  # pragma: no cover
        from difflib import SequenceMatcher

        return SequenceMatcher(None, e, a).find_longest_match(0, len(e), 0, len(a)).size >= threshold / 100 * len(e)


# ------------------------------------------------------------------ spec normalisation
def normalize_spec(spec: ExamSpec) -> tuple[ExamSpec, list[str]]:
    """Validate supplied marks without inventing or rescaling rubric allocations."""
    if not spec.questions:
        raise ValueError("No questions found in the question paper/spec.")
    seen: set[str] = set()
    for q in spec.questions:
        key = norm_qid(q.id)
        if not key or key in seen:
            raise ValueError(f"Question id is empty or duplicated: {q.id!r}.")
        seen.add(key)
        if not q.text.strip() or not q.model_answer.strip():
            raise ValueError(f"Q{q.id}: question and model answer are required.")
        if not math.isfinite(q.max_marks) or q.max_marks <= 0:
            raise ValueError(f"Q{q.id}: maximum marks must be finite and positive.")
        names = [c.name.strip().casefold() for c in q.criteria]
        if not q.criteria or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError(f"Q{q.id}: explicit, uniquely named rubric criteria are required.")
        for c in q.criteria:
            if not c.description.strip() or not math.isfinite(c.max_marks) or c.max_marks <= 0:
                raise ValueError(f"Q{q.id}, {c.name}: explicit positive criterion marks and description required.")
        total = sum(c.max_marks for c in q.criteria)
        if abs(total - q.max_marks) > 1e-6:
            raise ValueError(f"Q{q.id}: criterion marks sum to {total}, not {q.max_marks}. Fix the rubric.")
    return spec, []



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
        if not math.isfinite(cs.marks_awarded) or cs.marks_awarded < 0 or cs.marks_awarded > c.max_marks + 1e-6:
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
    """Compute one bounded score per criterion; preserve unresolved issues for review."""
    rubric = {c.name.strip().lower(): c for c in q.criteria}
    by_name = {}
    for cs in draft.criterion_scores:
        key = cs.criterion.strip().lower()
        if key in rubric and key not in by_name:
            by_name[key] = cs
    scores = []
    for key, c in rubric.items():
        cs = by_name.get(key)
        marks = min(max(cs.marks_awarded, 0), c.max_marks) if cs and math.isfinite(cs.marks_awarded) else 0.0
        if cs and marks > 0 and not evidence_found(cs.evidence, answer):
            marks = 0.0
        scores.append(AssessedCriterion(
            criterion=c.name, marks_awarded=marks, max_marks=c.max_marks,
            marks_withheld=round(c.max_marks - marks, 6), evidence=cs.evidence if cs and marks > 0 else "",
            justification=cs.justification if cs else c.description,
        ))
    total = round(sum(s.marks_awarded for s in scores), 6)
    return QuestionGrade(
        question_id=q.id, max_marks=q.max_marks, marks_awarded=total, attempted=True,
        criterion_scores=scores, strengths=draft.strengths, gaps=draft.gaps,
        how_to_improve=draft.how_to_improve, answer_text=answer, attempts=attempts,
        verified=not issues, verification_notes=issues,
    )



def build_unattempted(q: Question) -> QuestionGrade:
    return QuestionGrade(
        question_id=q.id, max_marks=q.max_marks, marks_awarded=0.0, attempted=False,
        criterion_scores=[
            AssessedCriterion(criterion=c.name, marks_awarded=0, max_marks=c.max_marks,
                              marks_withheld=c.max_marks, evidence="", justification=c.description)
            for c in q.criteria
        ],
    )
