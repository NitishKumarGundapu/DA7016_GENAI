"use strict";
const form = document.querySelector("#assessment-form");
const fields = document.querySelector("#form-fields");
const errorBox = document.querySelector("#error-message");
const steps = [...document.querySelectorAll("[data-node]")];
const status = document.querySelector("#result-status");
const progress = document.querySelector("#progress-message");
const studentInput = document.querySelector("#student-files");
const fileList = document.querySelector("#file-list");
let answerFiles = [];
let assessment = null;
let busy = false;

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function renderFiles() {
  fileList.replaceChildren();
  answerFiles.forEach((file, index) => {
    const row = element("li");
    row.append(element("span", `${index + 1}. ${file.name} · ${(file.size / 1024).toFixed(0)} KB`));
    for (const [label, offset] of [["Move up", -1], ["Move down", 1], ["Remove", 0]]) {
      const button = element("button", offset === -1 ? "↑" : offset === 1 ? "↓" : "×");
      button.type = "button";
      button.setAttribute("aria-label", `${label}: ${file.name}`);
      button.disabled = offset === -1 && index === 0 || offset === 1 && index === answerFiles.length - 1;
      button.addEventListener("click", () => {
        if (!offset) answerFiles.splice(index, 1);
        else [answerFiles[index], answerFiles[index + offset]] = [answerFiles[index + offset], answerFiles[index]];
        renderFiles();
      });
      row.append(button);
    }
    fileList.append(row);
  });
  studentInput.required = answerFiles.length === 0;
  if (!answerFiles.length) studentInput.value = "";
}

studentInput.addEventListener("change", () => {
  answerFiles = [...studentInput.files];
  renderFiles();
});
const drop = document.querySelector("#student-drop");
drop.addEventListener("dragover", event => { event.preventDefault(); if (!busy) drop.classList.add("dragging"); });
drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
drop.addEventListener("drop", event => {
  event.preventDefault();
  drop.classList.remove("dragging");
  if (!busy) { answerFiles = [...event.dataTransfer.files]; renderFiles(); }
});
document.querySelectorAll(".upload-card input").forEach(input => {
  input.addEventListener("change", () => {
    input.closest("label").querySelector(".file-name").textContent = input.files[0]?.name || "Choose a file ↗";
  });
});
for (const mode of ["file", "text"]) {
  document.querySelector(`#${mode}-mode`).addEventListener("click", () => {
    for (const candidate of ["file", "text"]) {
      const selected = mode === candidate;
      const button = document.querySelector(`#${candidate}-mode`);
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
      const input = document.querySelector(`#model-${candidate}`);
      input.hidden = !selected;
      input.disabled = !selected;
      input.required = selected;
    }
  });
}

function updateStep(index, state) {
  const step = steps[index];
  if (!step) return;
  step.className = state;
  step.querySelector(".step-state").textContent = {done: "Complete", running: "Working", failed: "Stopped", "": "Waiting"}[state];
  step.querySelector(".step-marker").textContent = state === "done" ? "✓" : index + 1;
}

function feedbackBlock(parent, title, text) {
  if (!text) return;
  const block = element("div", undefined, "feedback-block");
  block.dir = "auto";
  block.append(element("h4", title), element("p", text));
  parent.append(block);
}

function renderResults(data) {
  const result = data.result;
  document.querySelector("#empty-results").hidden = true;
  document.querySelector("#results").hidden = false;
  document.querySelector("#total-score").textContent = result.total;
  document.querySelector("#max-score").textContent = `/ ${result.max_total}`;
  document.querySelector("#percentage").textContent = `${result.percentage}%`;
  document.querySelector("#result-title").textContent = result.title;
  document.querySelector("#overall-summary").textContent = result.overall.summary;
  const warnings = document.querySelector("#warnings");
  warnings.replaceChildren(...(result.warnings || []).map(text => element("li", text)));
  const rows = document.querySelector("#grade-rows");
  const details = document.querySelector("#question-feedback");
  rows.replaceChildren();
  details.replaceChildren();
  for (const grade of result.grades) {
    const row = element("tr");
    const review = element("td");
    review.append(element("span", grade.verified ? "Verified" : "Review needed", `review-tag${grade.verified ? "" : " needs-review"}`));
    row.append(element("td", `${grade.question_id}${grade.attempted ? "" : " · Not attempted"}`), element("td", `${grade.marks_awarded} / ${grade.max_marks}`), review);
    rows.append(row);
    const detail = element("details", undefined, "question-detail");
    detail.append(element("summary", `Question ${grade.question_id} · ${grade.marks_awarded} / ${grade.max_marks}`));
    for (const criterion of grade.criterion_scores) {
      const card = element("div", undefined, "criterion");
      card.dir = "auto";
      card.append(element("strong", `${criterion.criterion} · ${criterion.marks_awarded} / ${criterion.max_marks}`));
      card.append(element("span", `Withheld: ${criterion.marks_withheld}`));
      if (criterion.evidence) card.append(element("blockquote", criterion.evidence));
      card.append(element("p", criterion.justification));
      detail.append(card);
    }
    feedbackBlock(detail, "Strengths", grade.strengths);
    feedbackBlock(detail, "Gaps", grade.gaps);
    feedbackBlock(detail, "How to improve", grade.how_to_improve);
    feedbackBlock(detail, "Verification notes", grade.verification_notes.join("\n"));
    details.append(detail);
  }
  feedbackBlock(details, "Study priorities", result.overall.top_priorities.join("\n"));
  document.querySelector("#transcript").textContent = data.transcript;
  document.querySelector("#run-info").textContent = data.tracing ? `LangSmith run: ${data.run_id}` : `Run: ${data.run_id} · LangSmith tracing is off.`;
}

function handleEvent(event) {
  if (event.type === "error") throw new Error(event.error);
  if (event.type === "progress") {
    const index = steps.findIndex(step => step.dataset.node === event.node);
    if (index >= 0) {
      updateStep(index, "done");
      updateStep(index + 1, "running");
      progress.textContent = `${index + 1} of ${steps.length} steps complete. ${steps[index + 1]?.querySelector("strong").textContent || "Finishing up"}…`;
    }
  } else if (event.type === "final") {
    assessment = event;
    renderResults(event);
    steps.forEach((_, index) => updateStep(index, "done"));
    status.textContent = "ASSESSMENT COMPLETE";
    status.className = "live-pill complete";
    progress.textContent = "Assessment complete. Review any flagged answers before using these marks.";
  }
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (busy) return;
  errorBox.hidden = true;
  const data = new FormData(form);
  data.delete("answer_files");
  answerFiles.forEach(file => data.append("answer_files", file));
  const files = [...data.values()].filter(value => value instanceof File && value.name);
  const supported = /\.(pdf|txt|md|png|jpe?g|webp|gif)$/i;
  if (!answerFiles.length) return showError("Add at least one student answer sheet.");
  if (files.some(file => !supported.test(file.name) || !file.size)) return showError("Choose nonempty PDF, text, or supported image files.");
  if (files.reduce((sum, file) => sum + file.size, 0) > 49 * 1024 * 1024) return showError("Files are too large. Keep uploads below 49 MB to allow room for request metadata.");
  busy = true;
  assessment = null;
  fields.disabled = true;
  document.querySelector("#results").hidden = true;
  document.querySelector("#empty-results").hidden = false;
  document.querySelector("#submit-button span").textContent = "Assessment in progress…";
  status.textContent = "ASSESSING";
  status.className = "live-pill running";
  steps.forEach((_, index) => updateStep(index, index === 0 ? "running" : ""));
  progress.textContent = "Reading your documents. Larger scans may take a little longer.";
  try {
    const response = await fetch("/api/assess", {method: "POST", body: data});
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || `Request failed (${response.status}).`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value, {stream: !done});
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) if (line.trim()) handleEvent(JSON.parse(line));
      if (done) break;
    }
    if (buffer.trim()) handleEvent(JSON.parse(buffer));
    if (!assessment) throw new Error("Connection ended before assessment completed. Please try again.");
  } catch (error) {
    showError(error.message);
    status.textContent = "NEEDS ATTENTION";
    status.className = "live-pill";
    steps.forEach((step, index) => { if (step.classList.contains("running")) updateStep(index, "failed"); });
    progress.textContent = "Assessment stopped. See the error beside your documents.";
  } finally {
    busy = false;
    fields.disabled = false;
    document.querySelector("#submit-button span").textContent = "Start assessment";
    renderFiles();
  }
});

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
window.addEventListener("beforeunload", event => { if (busy) event.preventDefault(); });
