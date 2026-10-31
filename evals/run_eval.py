"""LangSmith evaluation of the grader.

    python evals/run_eval.py                      # uses evals/dataset.json
    python evals/run_eval.py --dataset my-ds      # existing LangSmith dataset

dataset.json format:
[
  {"name": "student_a",
   "inputs":  {"question_paper": "examples/example_0_texts/question_paper.txt",
               "model_answers": "examples/example_0_texts/model_answers.txt",
               "rubric": "examples/example_0_texts/rubric.txt",
               "answer_files": ["examples/example_0_texts/student_answer.txt"]},
   "outputs": {"human_marks": {"1": 4, "2": 3}}}        # human_marks optional
]

Evaluators
  marks_within_bounds   every question 0..max
  total_consistency     total == sum(per-question marks)
  evidence_grounded     fraction of awarded criteria whose quote is truly in the answer
  all_verified          fraction of questions that passed verification
  mae_vs_human          1 - mean|AI - human| / max_marks   (only when human_marks given)
  feedback_quality      LLM-as-judge: is feedback specific + actionable? (0-1)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langsmith import Client, evaluate  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from auto_assessment.config import Settings  # noqa: E402
from auto_assessment.llm_client import create_llm  # noqa: E402
from auto_assessment.pipeline import run_assessment  # noqa: E402
from auto_assessment.verify import evidence_found  # noqa: E402


def target(inputs: dict) -> dict:
    """LangSmith target: run one assessment and keep the final result."""
    st = run_assessment(question_paper=inputs.get("question_paper"),
                        rubric=inputs.get("rubric"), model_answers=inputs.get("model_answers"),
                        model_answers_text=inputs.get("model_answers_text"),
                        answer_files=inputs["answer_files"], tags=["eval"])
    return {"result": st["result"]}


def marks_within_bounds(outputs: dict) -> dict:
    """Every question score must fall between zero and its maximum."""
    ok = all(0 <= g["marks_awarded"] <= g["max_marks"] for g in outputs["result"]["grades"])
    return {"key": "marks_within_bounds", "score": int(ok)}


def total_consistency(outputs: dict) -> dict:
    """The paper total must equal the sum of the per-question marks."""
    r = outputs["result"]
    return {"key": "total_consistency",
            "score": int(abs(r["total"] - sum(g["marks_awarded"] for g in r["grades"])) < 0.01)}


def evidence_grounded(outputs: dict) -> dict:
    """Fraction of awarded criteria whose quote really appears in the answer."""
    hits = tot = 0
    for g in outputs["result"]["grades"]:
        for c in g["criterion_scores"]:
            if c["marks_awarded"] > 0:
                tot += 1
                hits += evidence_found(c["evidence"], g["answer_text"])
    return {"key": "evidence_grounded", "score": hits / tot if tot else 1.0}


def all_verified(outputs: dict) -> dict:
    """Fraction of questions that passed automatic verification."""
    gs = outputs["result"]["grades"]
    return {"key": "all_verified", "score": sum(g["verified"] for g in gs) / max(len(gs), 1)}


def mae_vs_human(outputs: dict, reference_outputs: dict) -> dict:
    """Mean absolute error against optional human marks, normalized by question maxima."""
    human = (reference_outputs or {}).get("human_marks")
    if not human:
        return {"key": "mae_vs_human", "score": None}
    errs = [abs(g["marks_awarded"] - human[g["question_id"]]) / g["max_marks"]
            for g in outputs["result"]["grades"] if g["question_id"] in human]
    return {"key": "mae_vs_human", "score": 1 - sum(errs) / len(errs) if errs else None}


class _Judge(BaseModel):
    """Structured response from the feedback-quality judge."""

    score: int  # 1-5
    reason: str


def feedback_quality(outputs: dict) -> dict:
    """LLM-as-judge score for how specific and actionable the feedback is."""
    judge = create_llm(Settings.from_env()).structured(_Judge, temperature=0)
    scores = []
    for g in outputs["result"]["grades"]:
        if not g["attempted"]:
            continue
        v = judge.invoke(
            "Rate 1-5 how SPECIFIC and ACTIONABLE this exam feedback is for a student "
            "(5 = names exact missing points + concrete fix; 1 = generic).\n\n"
            f"Student answer:\n{g['answer_text']}\n\nMarks {g['marks_awarded']}/{g['max_marks']}\n"
            f"Strengths: {g['strengths']}\nGaps: {g['gaps']}\nHow to improve: {g['how_to_improve']}")
        scores.append((v.score - 1) / 4)
    return {"key": "feedback_quality", "score": sum(scores) / len(scores) if scores else None}


def ensure_dataset(client: Client, name: str, path: Path) -> str:
    """Create the LangSmith dataset from a local JSON file when it does not exist."""
    if client.has_dataset(dataset_name=name):
        return name
    rows = json.loads(path.read_text())
    ds = client.create_dataset(name, description="Auto-Assessment eval set")
    client.create_examples(dataset_id=ds.id, inputs=[r["inputs"] for r in rows],
                           outputs=[r.get("outputs", {}) for r in rows])
    return name


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="auto-assessment-evals")
    ap.add_argument("--file", default=str(Path(__file__).parent / "dataset.json"))
    a = ap.parse_args()
    client = Client()
    name = ensure_dataset(client, a.dataset, Path(a.file))
    evaluate(target, data=name,
             evaluators=[marks_within_bounds, total_consistency, evidence_grounded, all_verified,
                         mae_vs_human, feedback_quality],
             experiment_prefix="auto-assessment", max_concurrency=2, client=client)
