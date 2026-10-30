"""Markdown report renderer."""


def _cell(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def render_report(result: dict, transcript: str, warnings: list[str]) -> str:
    labels = result["overall"]["labels"]
    L: list[str] = []
    L.append(f"# {result['title']} - {labels['assessment_report']}\n")
    L.append(f"## {labels['total']}: **{result['total']} / {result['max_total']}** ({result['percentage']}%)\n")

    ov = result.get("overall")
    if ov:
        L.append(ov["summary"] + "\n")
        if ov.get("top_priorities"):
            L.append(f"**{labels['study_priorities']}**")
            L += [f"{i}. {p}" for i, p in enumerate(ov["top_priorities"], 1)]
            L.append("")

    L.append(f"## {labels['marks_by_question']}\n")
    L.append(f"| {labels['question']} | {labels['marks']} | {labels['attempted']} | {labels['verified']} |")
    L.append("|---|-------|-----------|----------------|")
    for g in result["grades"]:
        L.append(f"| {_cell(g['question_id'])} | {g['marks_awarded']} / {g['max_marks']} | "
                 f"{labels['yes'] if g['attempted'] else labels['no']} | "
                 f"{labels['yes'] if g['verified'] else labels['no']} |")
    L.append("")

    L.append(f"## {labels['detailed_feedback']}\n")
    for g in result["grades"]:
        L.append(f"### {labels['question']} {g['question_id']} - {g['marks_awarded']} / {g['max_marks']}\n")
        if g["criterion_scores"]:
            L.append(f"| {labels['criterion']} | {labels['marks']} | {labels['maximum']} | "
                     f"{labels['withheld']} | {labels['evidence']} | {labels['justification']} |")
            L.append("|-----------|-------|---------|----------|----------|-----|")
            for c in g["criterion_scores"]:
                L.append(f"| {_cell(c['criterion'])} | {c['marks_awarded']} | {c['max_marks']} | "
                         f"{c['marks_withheld']} | "
                         f"{_cell(c['evidence'])} | {_cell(c['justification'])} |")
            L.append("")
        if g["strengths"]:
            L.append(f"**{labels['strengths']}:** {g['strengths']}\n")
        if g["gaps"]:
            L.append(f"**{labels['gaps']}:** {g['gaps']}\n")
        if g["how_to_improve"]:
            L.append(f"**{labels['improvement']}:** {g['how_to_improve']}\n")
        if g["verification_notes"]:
            L.append(f"> {labels['verification_notes']}: " + "; ".join(g["verification_notes"]) + "\n")

    if warnings:
        L.append(f"## {labels['warnings']}\n")
        L += [f"- {w}" for w in warnings]
        L.append("")

    L.append(f"<details><summary>{labels['transcript']}</summary>\n")
    L.append("```\n" + transcript.strip() + "\n```\n")
    L.append("</details>")
    return "\n".join(L)
