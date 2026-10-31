"use strict";
const form = document.querySelector("#assessment-form");
const fields = document.querySelector("#form-fields");
const errorBox = document.querySelector("#error-message");
const steps = [...document.querySelectorAll("[data-node]")];
const status = document.querySelector("#result-status");
const progress = document.querySelector("#progress-message");
const studentInput = document.querySelector("#student-files");
const fileList = document.querySelector("#file-list");
const reviewsLink = document.querySelector("#reviews-link");
const clearButtons = [document.querySelector("#clear-button")];
const gradeProgress = document.querySelector("#grade-progress");
const gradeProgressTrack = document.querySelector("#grade-progress-track");
const gradeProgressFill = document.querySelector("#grade-progress-fill");
const gradeProgressText = document.querySelector("#grade-progress-text");
const REVIEWS_KEY = "autoAssessment.reviews.v1";
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
function setMode(mode) {
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
}
for (const mode of ["file", "text"]) {
  document.querySelector(`#${mode}-mode`).addEventListener("click", () => setMode(mode));
}

function updateStep(index, state) {
  const step = steps[index];
  if (!step) return;
  step.className = state;
  step.querySelector(".step-state").textContent = {done: "Complete", running: "Working", failed: "Stopped", "": "Waiting"}[state];
  step.querySelector(".step-marker").textContent = state === "done" ? "✓" : index + 1;
}

function updateGradeProgress(completed, total, questionId) {
  gradeProgress.hidden = !total;
  const percent = total ? Math.min(100, Math.round((100 * completed) / total)) : 0;
  gradeProgressFill.style.width = `${percent}%`;
  gradeProgressTrack.setAttribute("aria-valuemax", String(total || 0));
  gradeProgressTrack.setAttribute("aria-valuenow", String(completed));
  gradeProgressText.textContent = total
    ? (completed >= total
      ? `${total} of ${total} questions graded`
      : `Question ${questionId} · ${completed} of ${total} graded`)
    : "0 of 0 questions";
}

function resetGradeProgress() {
  gradeProgress.hidden = true;
  gradeProgressFill.style.width = "0%";
  gradeProgressTrack.setAttribute("aria-valuemax", "0");
  gradeProgressTrack.setAttribute("aria-valuenow", "0");
  gradeProgressText.textContent = "0 of 0 questions";
}

function storedAssessment() {
  try {
    const raw = sessionStorage.getItem(REVIEWS_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (error) {
    return null;
  }
}

function storeAssessment(data) {
  try {
    sessionStorage.setItem(REVIEWS_KEY, JSON.stringify({
      result: data.result, report_md: data.report_md, transcript: data.transcript,
      items: data.items, run_id: data.run_id, tracing: data.tracing,
      saved_at: new Date().toISOString(),
    }));
    return true;
  } catch (error) {
    console.warn("Could not keep the review sheet in this browser.", error);
    return false;
  }
}

function updateReviewsLink() {
  const available = Boolean(assessment && storedAssessment());
  reviewsLink.setAttribute("aria-disabled", String(!available));
  reviewsLink.tabIndex = available ? 0 : -1;
  reviewsLink.classList.toggle("disabled", !available);
  reviewsLink.classList.toggle("ready", available);
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
  } else if (event.type === "question") {
    updateGradeProgress(event.completed, event.total, event.question_id);
    if (event.completed < event.total) {
      progress.textContent = `Grading question ${event.completed + 1} of ${event.total} · ${event.completed} finished.`;
    }
  } else if (event.type === "final") {
    assessment = event;
    steps.forEach((_, index) => updateStep(index, "done"));
    status.textContent = "ASSESSMENT COMPLETE";
    status.className = "live-pill complete";
    progress.textContent = "Assessment complete. Review any flagged answers before using these marks.";
    const grades = event.result.grades || [];
    updateGradeProgress(grades.length, grades.length, "");
    if (!storeAssessment(event)) {
      progress.textContent = "Assessment complete, but the review sheet could not be stored in this browser (storage limit). Download the report instead.";
    }
    updateReviewsLink();
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
  clearButtons.forEach(button => { if (button) button.disabled = true; });
  sessionStorage.removeItem(REVIEWS_KEY);
  updateReviewsLink();
  resetGradeProgress();
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
    clearButtons.forEach(button => { if (button) button.disabled = false; });
    document.querySelector("#submit-button span").textContent = "Start assessment";
    renderFiles();
  }
});

window.addEventListener("beforeunload", event => { if (busy) event.preventDefault(); });

function clearAssessment() {
  sessionStorage.removeItem(REVIEWS_KEY);
  assessment = null;
  answerFiles = [];
  form.reset();
  setMode("file");
  document.querySelectorAll(".upload-card .file-name").forEach(label => {
    label.textContent = "Choose a file ↗";
  });
  renderFiles();
  document.querySelector("#error-message").hidden = true;
  steps.forEach((_, index) => updateStep(index, ""));
  status.textContent = "READY WHEN YOU ARE";
  status.className = "live-pill";
  progress.textContent = "A clear trail from first page to final feedback.";
  resetGradeProgress();
  updateReviewsLink();
  window.scrollTo({top: 0, behavior: "smooth"});
}

for (const button of clearButtons) {
  if (!button) continue;
  button.addEventListener("click", () => {
    if (busy) return;
    if (!confirm("Clear this assessment and start a new one? Saved downloads are not affected.")) return;
    clearAssessment();
  });
}

reviewsLink.addEventListener("click", event => {
  if (busy || !assessment || !storedAssessment()) event.preventDefault();
});

function restoreStoredAssessment() {
  const stored = storedAssessment();
  if (!stored || !stored.result) return;
  assessment = stored;
  steps.forEach((_, index) => updateStep(index, "done"));
  status.textContent = "ASSESSMENT COMPLETE";
  status.className = "live-pill complete";
  const grades = stored.result.grades || [];
  updateGradeProgress(grades.length, grades.length, "");
  progress.textContent = "Assessment complete. Review any flagged answers before using these marks.";
  updateReviewsLink();
}

restoreStoredAssessment();
updateReviewsLink();
