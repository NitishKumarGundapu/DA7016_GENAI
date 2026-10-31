"""Render the student-facing Markdown report."""
from __future__ import annotations


def _cell(value: str) -> str:
    """Escape a value for a Markdown table cell."""
    return (value or "").replace("|", "\\|").replace("\n", " ").strip()


def render_report(result: dict, transcript: str, warnings: list[str]) -> str:
    """Build the complete Markdown report from a final ``result`` mapping.

    Headings and status labels come from ``result["overall"]["labels"]`` so the
    report follows the language of the question paper.
    """
    labels = result["overall"]["labels"]
    lines: list[str] = []
    lines.append(f"# {result['title']} - {labels['assessment_report']}\n")
    lines.append(
        f"## {labels['total']}: **{result['total']} / {result['max_total']}** "
        f"({result['percentage']}%)\n"
    )

    overall = result.get("overall")
    if overall:
        lines.append(overall["summary"] + "\n")
        if overall.get("top_priorities"):
            lines.append(f"**{labels['study_priorities']}**")
            lines.extend(
                f"{index}. {priority}"
                for index, priority in enumerate(overall["top_priorities"], 1)
            )
            lines.append("")

    lines.append(f"## {labels['marks_by_question']}\n")
    lines.append(
        f"| {labels['question']} | {labels['marks']} | {labels['attempted']} | "
        f"{labels['verified']} |"
    )
    lines.append("|---|-------|-----------|----------------|")
    for grade in result["grades"]:
        lines.append(
            f"| {_cell(grade['question_id'])} | "
            f"{grade['marks_awarded']} / {grade['max_marks']} | "
            f"{labels['yes'] if grade['attempted'] else labels['no']} | "
            f"{labels['yes'] if grade['verified'] else labels['no']} |"
        )
    lines.append("")

    lines.append(f"## {labels['detailed_feedback']}\n")
    for grade in result["grades"]:
        lines.append(
            f"### {labels['question']} {grade['question_id']} - "
            f"{grade['marks_awarded']} / {grade['max_marks']}\n"
        )

        if grade["criterion_scores"]:
            lines.append(
                f"| {labels['criterion']} | {labels['marks']} | {labels['maximum']} | "
                f"{labels['withheld']} | {labels['evidence']} | {labels['justification']} |"
            )
            lines.append("|-----------|-------|---------|----------|----------|-----|")
            for criterion in grade["criterion_scores"]:
                lines.append(
                    f"| {_cell(criterion['criterion'])} | {criterion['marks_awarded']} | "
                    f"{criterion['max_marks']} | {criterion['marks_withheld']} | "
                    f"{_cell(criterion['evidence'])} | {_cell(criterion['justification'])} |"
                )
            lines.append("")

        if grade["strengths"]:
            lines.append(f"**{labels['strengths']}:** {grade['strengths']}\n")
        if grade["gaps"]:
            lines.append(f"**{labels['gaps']}:** {grade['gaps']}\n")
        if grade["how_to_improve"]:
            lines.append(f"**{labels['improvement']}:** {grade['how_to_improve']}\n")
        if grade["verification_notes"]:
            lines.append(
                f"> {labels['verification_notes']}: "
                + "; ".join(grade["verification_notes"])
                + "\n"
            )

    if warnings:
        lines.append(f"## {labels['warnings']}\n")
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    lines.append(f"<details><summary>{labels['transcript']}</summary>\n")
    lines.append("```\n" + transcript.strip() + "\n```\n")
    lines.append("</details>")
    return "\n".join(lines)
