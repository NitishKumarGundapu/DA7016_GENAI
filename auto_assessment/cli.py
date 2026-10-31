"""Command-line entry point for running one assessment and saving its artifacts.

Example:
    python -m auto_assessment.cli \\
        --question-paper qp.pdf --rubric rubric.pdf \\
        --model-answers model.pdf --answers page1.jpg page2.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings
from .pipeline import run_assessment, tracing_enabled


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser so it can be reused by tests and tooling."""
    parser = argparse.ArgumentParser(description="Auto-Assessment (LangGraph)")
    parser.add_argument("--question-paper", required=True)
    parser.add_argument("--rubric", help="Optional rubric document with explicit criterion marks")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model-answers", help="Model-answer document, PDF, or scan")
    model_group.add_argument(
        "--model-answers-text",
        help="Inline correct answers, with question labels",
    )

    parser.add_argument("--answers", nargs="+", required=True, help="Scans (png/jpg/pdf) or .txt")
    parser.add_argument("--provider", choices=["openrouter", "ollama"])
    parser.add_argument("--no-critic", action="store_true")
    parser.add_argument("--out", default="out", help="Directory for report.md and JSON outputs")
    return parser


def main() -> None:
    """Run an assessment from CLI arguments and write the report artifacts."""
    args = build_parser().parse_args()

    settings = Settings.from_env()
    if args.provider:
        settings.provider = args.provider
    if args.no_critic:
        settings.use_critic = False

    state = run_assessment(
        question_paper=args.question_paper,
        rubric=args.rubric,
        model_answers=args.model_answers,
        model_answers_text=args.model_answers_text,
        answer_files=args.answers,
        settings=settings,
    )

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "report.md": state["report_md"],
        "result.json": json.dumps(state["result"], indent=2, ensure_ascii=False),
        "questions.json": json.dumps(state["items"], indent=2, ensure_ascii=False),
        "transcript.txt": state["transcript"],
    }
    for filename, content in artifacts.items():
        (output_dir / filename).write_text(content, encoding="utf-8")

    result = state["result"]
    print(f"\nTotal: {result['total']} / {result['max_total']} ({result['percentage']}%)")
    for grade in result["grades"]:
        print(f"  Q{grade['question_id']}: {grade['marks_awarded']}/{grade['max_marks']}")
    tracing = "on" if tracing_enabled() else "off"
    print(
        f"\nSaved to {output_dir}/  |  LangSmith tracing: {tracing}"
        f"  |  run_id={state['run_id']}"
    )


if __name__ == "__main__":
    main()
