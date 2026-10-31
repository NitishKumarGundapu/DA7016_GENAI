"""Run the bundled example, or forward CLI arguments to ``auto_assessment.cli``.

Examples:
    python run_pipeline.py
    python run_pipeline.py --question-paper qp.pdf --model-answers m.pdf --answers a.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

from auto_assessment.cli import main as cli_main


def main() -> None:
    """Use the example inputs when no arguments are supplied, then run the CLI."""
    if len(sys.argv) == 1:
        root = Path(__file__).resolve().parent
        sys.argv.extend([
            "--question-paper", str(root / "examples" / "example_0_texts" / "question_paper.txt"),
            "--model-answers", str(root / "examples" / "example_0_texts" / "model_answers.txt"),
            "--answers", str(root / "examples" / "example_0_texts" / "student_answer.txt"),
            "--out", str(root / "out" / "pipeline-test"),
        ])
    cli_main()


if __name__ == "__main__":
    main()
