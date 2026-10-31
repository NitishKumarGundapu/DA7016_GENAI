"use strict";
const REVIEWS_KEY = "autoAssessment.reviews.v1";
const stored = readStored();
let assessment = stored;

function readStored() {
  try {
    const raw = sessionStorage.getItem(REVIEWS_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (error) {
    return null;
  }
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function feedbackBlock(parent, title, text) {
  if (!text) return;
  const block = element("div", undefined, "feedback-block");
  block.dir = "auto";
  block.append(element("h4", title), element("p", text));
  parent.append(block);
}

function renderCard(grade) {
  const card = element("article", undefined, "review-card");
  card.dir = "auto";
  const head = element("div", undefined, "review-card-head");
  const label = element("div", undefined, "review-card-label");
  label.append(element("h3", `Question ${grade.question_id}${grade.attempted ? "" : " · Not attempted"}`));
  label.append(element("span", grade.verified ? "Evidence verified by code and critic" : "Human review recommended", "review-tag" + (grade.verified ? "" : " needs-review")));
  const marks = element("div", undefined, "review-marks");
  marks.append(document.createTextNode(`${grade.marks_awarded}`), element("small", `/ ${grade.max_marks}`));
  head.append(label, marks);
  card.append(head);
  if (!grade.attempted) card.append(element("p", "No answer was found for this question, so it was not sent for grading.", "review-meta"));
  for (const criterion of grade.criterion_scores || []) {
    const block = element("section", undefined, "criterion");
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
    card.append(block);
  }
  feedbackBlock(card, "Strengths", grade.strengths);
  feedbackBlock(card, "Gaps", grade.gaps);
  feedbackBlock(card, "How to improve", grade.how_to_improve);
  const notes = grade.verification_notes || [];
  if (notes.length) feedbackBlock(card, "Verification notes", notes.join("\n"));
  return card;
}

function render(data) {
  const cards = document.querySelector("#review-cards");
  cards.replaceChildren(...(data.result.grades || []).map(renderCard));
  if (!cards.children.length) cards.append(element("p", "No questions were graded in this run.", "review-meta"));
  document.querySelector("#review-transcript").textContent = data.transcript || "";
  document.querySelector("#reviews-empty").hidden = true;
  document.querySelector("#reviews-content").hidden = false;
}

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

document.querySelector("#download-report").addEventListener("click", () => {
  if (assessment) download(assessment.report_md, "report.md", "text/markdown;charset=utf-8");
});
document.querySelector("#download-json").addEventListener("click", () => {
  if (assessment) download(JSON.stringify(assessment.result, null, 2), "result.json", "application/json");
});

if (assessment && assessment.result) {
  render(assessment);
} else {
  document.querySelector("#reviews-empty").hidden = false;
}
