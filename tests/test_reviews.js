// Run with node --test tests/test_reviews.js. No browser packages required.
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");
const source = fs.readFileSync(path.join(__dirname, "../static/reviews.js"), "utf8");

function load(raw = null) {
  const nodes = new Map();
  const windowEvents = {};
  function node(tag = "div") {
    return {
      tag, children: [], value: "", hidden: true, events: {}, isConnected: true,
      append(...children) { this.children.push(...children); },
      replaceChildren(...children) { this.children = children; },
      addEventListener(event, handler) { this.events[event] = handler; },
      querySelector(tag) { return this.children.find(child => child.tag === tag); },
      focus() {},
    };
  }
  const get = selector => {
    if (!nodes.has(selector)) nodes.set(selector, node());
    return nodes.get(selector);
  };
  get("#review-filter").value = "all";
  get("#review-sort").value = "paper";
  const context = vm.createContext({
    document: {
      querySelector: get, getElementById: id => get(`#${id}`),
      createElement: node, createTextNode: text => ({textContent: text}),
      querySelectorAll: () => get("#review-cards").children,
    },
    sessionStorage: {getItem: () => raw},
    window: {addEventListener(event, handler) { windowEvents[event] = handler; }},
    console: {warn() {}},
  });
  vm.runInContext(source, context);
  return {context, get, windowEvents, run: code => vm.runInContext(code, context)};
}

const grades = [
  {question_id: "10", marks_awarded: 1, max_marks: 4, attempted: true, verified: false,
    answer_text: "chlorophyll", verification_notes: ["Check evidence"]},
  {question_id: "2", marks_awarded: 0, max_marks: 2, attempted: false, verified: true},
  {question_id: "1", marks_awarded: 2, max_marks: 2, attempted: true, verified: true},
];
const data = {
  result: {title: "Toy assessment", total: 3, max_total: 8, percentage: 37.5, grades,
    overall: {summary: "Review Q10, attempt Q2.", top_priorities: ["Practise Q2."]},
    warnings: ["Q10 needs human review."]},
  items: [{question_id: "10", question: "Explain photosynthesis", answer: "chlorophyll",
    model_answer: "Plants use sunlight.", rubric: "Scientific explanation"}],
};

test("paper order, natural labels, status filters, score sorts and aligned search", () => {
  const {run} = load(JSON.stringify(data));
  const ids = (search, filter, sort) => JSON.parse(run(
    `JSON.stringify(visibleGrades(assessment, ${JSON.stringify(search)}, ${JSON.stringify(filter)}, ${JSON.stringify(sort)}).map(row => row.grade.question_id))`,
  ));
  assert.deepEqual(ids("", "all", "paper"), ["10", "2", "1"]);
  assert.deepEqual(ids("", "all", "question"), ["1", "2", "10"]);
  assert.deepEqual(ids("", "all", "needs-review"), ["10", "2", "1"]);
  assert.deepEqual(ids("", "all", "withheld"), ["10", "2", "1"]);
  assert.deepEqual(ids("", "all", "lowest"), ["2", "10", "1"]);
  assert.deepEqual(ids("", "all", "highest"), ["1", "10", "2"]);
  assert.deepEqual(ids("", "needs-review", "paper"), ["10"]);
  assert.deepEqual(ids("", "unattempted", "paper"), ["2"]);
  assert.deepEqual(ids("", "verified", "paper"), ["2", "1"]);
  assert.deepEqual(ids("", "lost-marks", "paper"), ["10", "2"]);
  for (const query of [" PHOTOSYNTHESIS ", "sunlight", "scientific", "chlorophyll", "Check evidence"]) {
    assert.deepEqual(ids(query, "all", "paper"), ["10"]);
  }
  assert.deepEqual(ids("sunlight", "verified", "paper"), []);
  assert.equal(run("assessment.result.grades[0].question_id"), "10");
});

test("overview, warning visibility, empty searches and older saved runs", () => {
  const {get, run} = load(JSON.stringify(data));
  assert.equal(get("#review-title").textContent, data.result.title);
  assert.equal(get("#review-summary").textContent, data.result.overall.summary);
  assert.equal(get("#review-stats").children[3].children[0].textContent, "1");
  assert.equal(get("#review-warnings").hidden, false);
  assert.equal(get("#review-count").textContent, "Showing 3 of 3 questions");
  get("#review-search").value = "no-match";
  run("renderQuestions()");
  assert.equal(get("#review-no-matches").hidden, false);
  assert.equal(get("#print-reviews").disabled, true);
  const old = load(JSON.stringify({result: {grades: []}}));
  assert.equal(old.get("#reviews-content").hidden, false);
  assert.equal(old.get("#review-warnings").hidden, true);
});

test("CSV includes all scores, quotes text, preserves Unicode and blocks formulas", () => {
  const {run} = load(JSON.stringify(data));
  const csv = run("scoresCSV(assessment.result.grades)");
  assert.ok(csv.startsWith("\uFEFF"));
  assert.equal(csv.split("\r\n").length, 5);
  assert.ok(csv.includes('"10","1","4","3","25","Yes","Needs human review","Check evidence"'));
  const unsafe = {...grades[0], question_id: '=HYPERLINK("bad")',
    verification_notes: ["\t=1+1", "परीक्षा"]};
  assert.ok(run(`scoresCSV([${JSON.stringify(unsafe)}])`).includes(`"'=HYPERLINK(""bad"")"`));
  assert.ok(run(`scoresCSV([${JSON.stringify(unsafe)}])`).includes(`"'\t=1+1; परीक्षा"`));
});

test("expansion survives sorting and printing restores the previous open state", () => {
  const {get, run, windowEvents} = load(JSON.stringify(data));
  get("#expand-reviews").events.click();
  get("#review-sort").value = "question";
  run("renderQuestions()");
  assert.ok(get("#review-cards").children.every(card => card.open));
  get("#collapse-reviews").events.click();
  assert.ok(get("#review-cards").children.every(card => !card.open));
  windowEvents.beforeprint();
  assert.ok(get("#review-cards").children.every(card => card.open));
  windowEvents.afterprint();
  assert.ok(get("#review-cards").children.every(card => !card.open));
  assert.equal(run("openQuestions.size"), 0);
});

test("missing, corrupt and malformed storage displays explicit recoverable states", () => {
  assert.equal(load().get("#reviews-empty").hidden, false);
  for (const raw of ["{broken", '{"result":{}}', "null"]) {
    const {get} = load(raw);
    assert.equal(get("#review-error").hidden, false);
    assert.ok(get("#review-error").textContent.includes("Could not read"));
    assert.equal(get("#reviews-empty").hidden, false);
  }
});
