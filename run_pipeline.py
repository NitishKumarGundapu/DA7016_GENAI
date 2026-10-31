"""Run the example assessment, or pass the same arguments as auto_assessment.cli."""
from __future__ import annotations

import sys
from pathlib import Path

from auto_assessment.cli import main as cli_main


def main() -> None:
    if len(sys.argv) == 1:
        root = Path(__file__).resolve().parent
        sys.argv.extend([
            "--question-paper", str(root / "examples" / "question_paper.txt"),
            "--model-answers", str(root / "examples" / "model_answers.txt"),
            "--answers", str(root / "examples" / "student_answer.txt"),
            "--out", str(root / "out" / "pipeline-test"),
        ])
    cli_main()


if __name__ == "__main__":
    main()
