import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auto_assessment.schemas import Criterion, CriterionScore, ExamSpec, GradeDraft, Question
from auto_assessment.verify import build_grade, check_draft, evidence_found, norm_qid, normalize_spec

Q = Question(id="1", text="t", max_marks=3, model_answer="m", criteria=[
    Criterion(name="A", description="a", max_marks=1), Criterion(name="B", description="b", max_marks=2)])
ANS = "Overfitting means the model memorises noise and performs badly on unseen data."


def draft(a, b, ev_a="memorises noise", ev_b="performs badly on unseen data"):
    return GradeDraft(criterion_scores=[
        CriterionScore(criterion="A", marks_awarded=a, evidence=ev_a, justification="j"),
        CriterionScore(criterion="B", marks_awarded=b, evidence=ev_b, justification="j")],
        strengths="s", gaps="g", how_to_improve="h")


def test_norm_qid():
    assert norm_qid("Q2(a)") == norm_qid("2 a") == "2a"


def test_evidence():
    assert evidence_found("memorises noise", ANS)
    assert not evidence_found("gradient descent converges", ANS)


def test_clean_draft_has_no_issues():
    assert check_draft(Q, ANS, draft(1, 2)) == []


def test_out_of_bounds_and_fake_evidence():
    issues = check_draft(Q, ANS, draft(2, 2, ev_a="totally invented quote about gradients"))
    assert any("outside" in i for i in issues)
    assert any("not found" in i for i in issues)


def test_build_grade_clamps_and_sums_in_code():
    g = build_grade(Q, ANS, draft(5, 2), attempts=1, issues=[])
    assert g.marks_awarded == 3 and g.criterion_scores[0].marks_awarded == 1


def test_normalize_spec_rescales():
    q = Q.model_copy(deep=True)
    q.criteria[0].max_marks = 2  # sum 4 != 3
    spec, warns = normalize_spec(ExamSpec(title="x", questions=[q]))
    assert warns and abs(sum(c.max_marks for c in spec.questions[0].criteria) - 3) < 0.05
