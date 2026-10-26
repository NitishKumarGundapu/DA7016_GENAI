"""python -m auto_assessment.cli --question-paper qp.pdf --rubric r.pdf --model-answers m.pdf --answers p1.jpg p2.jpg"""
import argparse
import json
from pathlib import Path

from .config import Settings
from .pipeline import run_assessment, tracing_enabled


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-Assessment (LangGraph)")
    ap.add_argument("--question-paper")
    ap.add_argument("--rubric")
    ap.add_argument("--model-answers")
    ap.add_argument("--spec", help="Optional structured JSON spec (replaces the three documents)")
    ap.add_argument("--answers", nargs="+", required=True, help="Scans (png/jpg/pdf) or .txt")
    ap.add_argument("--provider", choices=["openai", "ollama", "openai_then_ollama", "ollama_then_openai"])
    ap.add_argument("--no-critic", action="store_true")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    s = Settings.from_env()
    if a.provider:
        s.provider = a.provider
    if a.no_critic:
        s.use_critic = False

    state = run_assessment(question_paper=a.question_paper, rubric=a.rubric, model_answers=a.model_answers,
                           spec_file=a.spec, answer_files=a.answers, settings=s)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(state["report_md"], encoding="utf-8")
    (out / "result.json").write_text(json.dumps(state["result"], indent=2), encoding="utf-8")
    (out / "transcript.txt").write_text(state["transcript"], encoding="utf-8")

    r = state["result"]
    print(f"\nTotal: {r['total']} / {r['max_total']} ({r['percentage']}%)")
    for g in r["grades"]:
        print(f"  Q{g['question_id']}: {g['marks_awarded']}/{g['max_marks']}")
    print(f"\nSaved to {out}/  |  LangSmith tracing: {'on' if tracing_enabled() else 'off'}  |  run_id={state['run_id']}")


if __name__ == "__main__":
    main()
