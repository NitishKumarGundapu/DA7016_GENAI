/**
 * Review-sheet controller.
 *
 * Reads the completed assessment from sessionStorage (written by app.js), then
 * renders the overview, filters, jump list, question cards, and downloads.
 */
"use strict";
const REVIEWS_KEY = "autoAssessment.reviews.v1";
const assessment = readStored();
const items = new Map((assessment?.items || []).map(item => [item.question_id, item]));
const openQuestions = new Set();
const numberOrder = new Intl.Collator(undefined, {numeric: true, sensitivity: "base"});

/** Show a recoverable storage/load error at the top of the sheet. */
function showError(message) {
  const errorBox = document.querySelector("#review-error");
  errorBox.textContent = message;
  errorBox.hidden = false;
}

/** Parse the saved assessment and validate its minimum shape. */
function readStored() {
  try {
    const raw = sessionStorage.getItem(REVIEWS_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (!data?.result || !Array.isArray(data.result.grades)) {
      throw new Error("Saved assessment is missing question grades.");
    }
    return data;
  } catch (error) {
    console.warn("Could not load the review sheet.", error);
    showError("Could not read the saved assessment. Return to the studio and run it again.");
    return null;
  }
}

/** Create a small DOM node with optional text and class name. */
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

/** Append a titled feedback block, skipping empty text. */
function feedbackBlock(parent, title, text) {
  if (!text) return;
  const block = element("section", undefined, "feedback-block");
  block.dir = "auto";
  block.append(element("h4", title), element("p", text));
  parent.append(block);
}

/** Human-readable verification/attempt status for a question. */
function statusText(grade) {
  if (!grade.attempted) return grade.verified ? "Not attempted" : "Not attempted · Needs human review";
  return grade.verified ? "Verification passed" : "Needs human review";
}

/** Fraction of available marks earned (0 when the question has no marks). */
function scorePercent(grade) {
  return grade.max_marks > 0 ? grade.marks_awarded / grade.max_marks : 0;
}

/** Apply the search box, filter, and sort order to the grade list. */
function visibleGrades(data, search, filter, sort) {
  const query = search.trim().toLocaleLowerCase();
  return data.result.grades.map((grade, index) => ({grade, index})).filter(({grade}) => {
    const item = items.get(grade.question_id) || {};
    const searchable = [
      grade.question_id, item.question, item.answer, item.model_answer, item.rubric,
      grade.answer_text, grade.strengths, grade.gaps, grade.how_to_improve,
      ...(grade.verification_notes || []),
      ...(grade.criterion_scores || []).flatMap(score => [score.criterion, score.evidence, score.justification]),
    ].filter(value => value != null).join("\n").toLocaleLowerCase();
    const matches = {
      all: true,
      "needs-review": !grade.verified,
      unattempted: !grade.attempted,
      verified: grade.verified,
      "lost-marks": grade.marks_awarded < grade.max_marks,
    };
    return matches[filter] && searchable.includes(query);
  }).sort((a, b) => {
    const comparisons = {
      paper: a.index - b.index,
      question: numberOrder.compare(a.grade.question_id, b.grade.question_id),
      "needs-review": Number(a.grade.verified) - Number(b.grade.verified),
      withheld: (b.grade.max_marks - b.grade.marks_awarded) - (a.grade.max_marks - a.grade.marks_awarded),
      lowest: scorePercent(a.grade) - scorePercent(b.grade),
      highest: scorePercent(b.grade) - scorePercent(a.grade),
    };
    return comparisons[sort] || a.index - b.index;
  });
}

/** Build one expandable question card, including criterion evidence. */
function renderCard({grade, index}) {
  const item = items.get(grade.question_id) || {};
  const card = element("details", undefined, "review-card");
  card.id = `review-question-${index}`;
  card.open = openQuestions.has(index);
  card.addEventListener("toggle", () => {
    if (!card.isConnected) return;
    if (card.open) openQuestions.add(index);
    else openQuestions.delete(index);
  });
  const head = element("summary", undefined, "review-card-head");
  const label = element("div", undefined, "review-card-label");
  label.append(element("h3", `Question ${grade.question_id}`));
  if (item.question) {
    const preview = element("span", item.question, "review-question-preview");
    preview.dir = "auto";
    label.append(preview);
  }
  label.append(element("span", statusText(grade), "review-tag" + (grade.verified ? "" : " needs-review")));
  const marks = element("div", undefined, "review-marks");
  marks.append(document.createTextNode(`${grade.marks_awarded}`), element("small", `/ ${grade.max_marks}`));
  head.append(label, marks);
  card.append(head);
  const body = element("div", undefined, "review-card-body");
  if (!grade.verified) {
    const warning = element("section", undefined, "review-alert");
    warning.append(element("h4", "Human review required"));
    const notes = element("ul");
    (grade.verification_notes?.length ? grade.verification_notes : ["Check this answer and its evidence before using the marks."])
      .forEach(note => notes.append(element("li", note)));
    warning.append(notes);
    body.append(warning);
  } else if (grade.verification_notes?.length) {
    feedbackBlock(body, "Verification notes", grade.verification_notes.join("\n"));
  }
  feedbackBlock(body, "Question", item.question);
  const comparison = element("div", undefined, "review-comparison");
  feedbackBlock(comparison, "Student answer", grade.attempted
    ? (item.answer || grade.answer_text || "Student answer unavailable in this saved run.")
    : "Not attempted. No answer was found; this question was not sent for grading.");
  feedbackBlock(comparison, "Model answer", item.model_answer || "Model answer unavailable in this saved run.");
  body.append(comparison);
  const feedback = element("div", undefined, "review-feedback");
  feedbackBlock(feedback, "Strengths", grade.strengths);
  feedbackBlock(feedback, "Gaps · Marks withheld", grade.gaps);
  feedbackBlock(feedback, "How to improve", grade.how_to_improve);
  body.append(feedback);
  if (grade.criterion_scores?.length) body.append(element("h4", "Criterion breakdown", "review-breakdown-title"));
  for (const criterion of grade.criterion_scores || []) {
    const block = element("section", undefined, "criterion");
    block.dir = "auto";
    const heading = element("div", undefined, "criterion-heading");
    heading.append(element("h4", criterion.criterion));
    const score = element("div", undefined, "criterion-score");
    score.append(
      element("strong", `${criterion.marks_awarded} / ${criterion.max_marks}`),
      element("span", `${criterion.marks_withheld} withheld`, "criterion-withheld"),
    );
    heading.append(score);
    block.append(heading);
    if (criterion.evidence) {
      const evidence = element("div", undefined, "criterion-evidence");
      evidence.dir = "auto";
      evidence.append(element("span", "Student evidence", "criterion-label"), element("blockquote", criterion.evidence));
      block.append(evidence);
    }
    const explanation = element("div", undefined, "criterion-explanation");
    explanation.dir = "auto";
    explanation.append(element("span", "Why these marks", "criterion-label"), element("p", criterion.justification));
    block.append(explanation);
    body.append(block);
  }
  card.append(body);
  return card;
}

/** Re-render the card list and jump list for the current controls. */
function renderQuestions() {
  if (!assessment) return;
  const rows = visibleGrades(assessment, document.querySelector("#review-search").value,
    document.querySelector("#review-filter").value, document.querySelector("#review-sort").value);
  document.querySelector("#review-cards").replaceChildren(...rows.map(renderCard));
  const navigation = document.querySelector("#review-question-nav");
  navigation.replaceChildren();
  for (const {grade, index} of rows) {
    const link = element("a", undefined, "review-question-link");
    link.href = `#review-question-${index}`;
    link.append(element("strong", `Q${grade.question_id}`),
      element("span", `${grade.marks_awarded} / ${grade.max_marks}`),
      element("small", statusText(grade), grade.verified ? "" : "needs-review"));
    link.addEventListener("click", () => {
      const card = document.getElementById(`review-question-${index}`);
      card.open = true;
      openQuestions.add(index);
      card.querySelector("summary").focus({preventScroll: true});
    });
    navigation.append(link);
  }
  document.querySelector("#review-count").textContent = `Showing ${rows.length} of ${assessment.result.grades.length} questions`;
  document.querySelector("#review-no-matches").hidden = rows.length !== 0;
  for (const id of ["expand-reviews", "collapse-reviews", "print-reviews"]) {
    document.getElementById(id).disabled = rows.length === 0;
  }
}

/** Render the overview panel and initial question list. */
function render(data) {
  const result = data.result;
  const grades = result.grades;
  document.querySelector("#review-title").textContent = result.title || "Assessment overview";
  const saved = new Date(data.saved_at);
  document.querySelector("#review-saved").textContent = Number.isNaN(saved.getTime())
    ? "Latest assessment in this browser tab" : `Assessed ${saved.toLocaleString()}`;
  const stats = [
    [`${result.total} / ${result.max_total}`, "Total marks"],
    [`${result.percentage}%`, "Overall score"],
    [`${grades.filter(grade => grade.attempted).length} / ${grades.length}`, "Questions attempted"],
    [String(grades.filter(grade => !grade.verified).length), "Need human review"],
  ];
  document.querySelector("#review-stats").replaceChildren(...stats.map(([value, label], index) => {
    const stat = element("div", undefined, "review-stat" + (index === 3 && value !== "0" ? " flagged" : ""));
    stat.append(element("strong", value), element("span", label));
    return stat;
  }));
  document.querySelector("#review-summary").textContent = result.overall?.summary || "No overall feedback was saved for this assessment.";
  const priorities = result.overall?.top_priorities || [];
  document.querySelector("#review-priorities").replaceChildren(
    ...priorities.map(text => element("li", text)),
  );
  if (!priorities.length) document.querySelector("#review-priorities").append(element("li", "No study priorities provided."));
  const warnings = result.warnings || [];
  document.querySelector("#review-warnings").hidden = warnings.length === 0;
  document.querySelector("#review-warning-list").replaceChildren(...warnings.map(text => element("li", text)));
  document.querySelector("#review-transcript").textContent = data.transcript || "No transcript was saved.";
  document.querySelector("#reviews-empty").hidden = true;
  document.querySelector("#reviews-content").hidden = false;
  renderQuestions();
}

/** Trigger a client-side file download without a server round trip. */
function download(content, name, type) {
  const url = URL.createObjectURL(new Blob([content], {type}));
  const link = element("a");
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Build a spreadsheet-safe CSV of every question score. */
function scoresCSV(grades) {
  const rows = [
    ["Question", "Awarded", "Maximum", "Withheld", "Score (%)", "Attempted", "Verification", "Notes"],
    ...grades.map(grade => [
      grade.question_id, grade.marks_awarded, grade.max_marks,
      grade.max_marks - grade.marks_awarded, Math.round(scorePercent(grade) * 1000) / 10,
      grade.attempted ? "Yes" : "No", statusText(grade), (grade.verification_notes || []).join("; "),
    ]),
  ];
  return "\uFEFF" + rows.map(row => row.map(value => {
    let text = String(value);
    // ponytail: neutralize spreadsheet formulas; keep exports dependency-free.
    if (typeof value === "string" && (/^\s*[=+\-@]/.test(text) || /^[\t\r\n]/.test(text))) text = "'" + text;
    return `"${text.replaceAll('"', '""')}"`;
  }).join(",")).join("\r\n") + "\r\n";
}

document.querySelector("#review-search").addEventListener("input", renderQuestions);
for (const id of ["review-filter", "review-sort"]) {
  document.getElementById(id).addEventListener("change", renderQuestions);
}
document.querySelector("#reset-reviews").addEventListener("click", () => {
  document.querySelector("#review-search").value = "";
  document.querySelector("#review-filter").value = "all";
  document.querySelector("#review-sort").value = "paper";
  renderQuestions();
});
for (const [id, open] of [["expand-reviews", true], ["collapse-reviews", false]]) {
  document.getElementById(id).addEventListener("click", () => {
    document.querySelectorAll(".review-card").forEach(card => {
      card.open = open;
      const index = Number(card.id.replace("review-question-", ""));
      if (open) openQuestions.add(index);
      else openQuestions.delete(index);
    });
  });
}
document.querySelector("#download-report").addEventListener("click", () => {
  if (assessment) download(assessment.report_md, "report.md", "text/markdown;charset=utf-8");
});
document.querySelector("#download-json").addEventListener("click", () => {
  if (assessment) download(JSON.stringify(assessment.result, null, 2), "result.json", "application/json");
});
document.querySelector("#download-csv").addEventListener("click", () => {
  if (assessment) download(scoresCSV(assessment.result.grades), "scores.csv", "text/csv;charset=utf-8");
});
let printState = [];
window.addEventListener("beforeprint", () => {
  printState = [...document.querySelectorAll(".review-card")].map(card => [card, card.open]);
  printState.forEach(([card]) => { card.open = true; });
});
window.addEventListener("afterprint", () => {
  printState.forEach(([card, open]) => { card.open = open; });
  printState = [];
});
document.querySelector("#print-reviews").addEventListener("click", () => window.print());

if (assessment) render(assessment);
else document.querySelector("#reviews-empty").hidden = false;
