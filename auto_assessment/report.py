"""Markdown report renderer."""


def _cell(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def render_report(result: dict, transcript: str, warnings: list[str]) -> str:
    L: list[str] = []
    L.append(f"# {result['title']} - Assessment Report\n")
    L.append(f"## Total: **{result['total']} / {result['max_total']}** ({result['percentage']}%)\n")

    ov = result.get("overall")
    if ov:
        L.append(ov["summary"] + "\n")
        if ov.get("top_priorities"):
            L.append("**Study priorities**")
            L += [f"{i}. {p}" for i, p in enumerate(ov["top_priorities"], 1)]
            L.append("")

    L.append("## Marks by question\n")
    L.append("| Q | Marks | Attempted | Fully verified |")
    L.append("|---|-------|-----------|----------------|")
    for g in result["grades"]:
        L.append(f"| {g['question_id']} | {g['marks_awarded']} / {g['max_marks']} | "
                 f"{'yes' if g['attempted'] else 'no'} | {'yes' if g['verified'] else 'no'} |")
    L.append("")

    L.append("## Detailed feedback\n")
    for g in result["grades"]:
        L.append(f"### Q{g['question_id']} - {g['marks_awarded']} / {g['max_marks']}\n")
        if g["criterion_scores"]:
            L.append("| Criterion | Marks | Evidence from your answer | Why |")
            L.append("|-----------|-------|---------------------------|-----|")
            for c in g["criterion_scores"]:
                L.append(f"| {_cell(c['criterion'])} | {c['marks_awarded']} | "
                         f"{_cell(c['evidence'])} | {_cell(c['justification'])} |")
            L.append("")
        if g["strengths"]:
            L.append(f"**What you did well:** {g['strengths']}\n")
        if g["gaps"]:
            L.append(f"**Where marks were lost:** {g['gaps']}\n")
        if g["how_to_improve"]:
            L.append(f"**How to improve:** {g['how_to_improve']}\n")
        if g["verification_notes"]:
            L.append("> Verification notes: " + "; ".join(g["verification_notes"]) + "\n")

    if warnings:
        L.append("## Pipeline warnings\n")
        L += [f"- {w}" for w in warnings]
        L.append("")

    L.append("<details><summary>Transcript of the answer sheet</summary>\n")
    L.append("```\n" + transcript.strip() + "\n```\n")
    L.append("</details>")
    return "\n".join(L)
