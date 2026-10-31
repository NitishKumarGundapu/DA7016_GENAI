# Auto-Assessment: agentic grading of handwritten answer sheets

LangGraph (orchestration) + LangChain (grading models and structured output) + LangSmith (tracing and evals).

**Required input:** question paper (with maximum marks), model answers per question, student answer sheets
**Optional input:** rubric with explicit criterion marks
**Output:** marks per question + rubric-defensible, student-specific feedback (Markdown report + JSON)

## Architecture

```
START -> ingest -> parse_spec -> clean_align -> grade_question -> aggregate -> report -> END
                                               |
                                               v (sequential, in question-paper order)
                              grade -> verify -> [issues? retry with critique] -> finalize
```

| Node | What it does |
|---|---|
| `ingest` | Reads UTF-8 text directly. Sends each complete PDF as a base64 file to OpenRouter's `file-parser` plugin with `mistral-ocr`; sends original images directly to the OpenRouter vision model. No local PDF parsing, page rendering, resizing, or image conversion. Preserves upload order and requests original page order within each document. |
| `parse_spec` | Aligns question paper, optional rubric, and model answers into an internal `ExamSpec`. Explicit positive criterion allocations must sum to each question total; questions with no extracted allocations use one whole-question criterion based on the model answer and printed maximum marks, with a report warning. Partial or inconsistent allocations and missing/invalid question totals stop assessment before grading. |
| `clean_align` | Builds `items`, a list of `{question_id, question, answer, model_answer, rubric, max_marks}` dictionaries. Student answers remain verbatim. Missing/duplicate mappings and ungrounded text skip only the affected question with withheld marks and a human-review flag, rather than stopping the whole assessment or silently becoming zero marks; unknown ids are ignored with a warning. |
| `grade_question` | Evaluates questions one by one in paper order, using each aligned dictionary as input. A question flagged during alignment or whose grading raises is skipped with withheld marks and a review warning; the remaining questions still run. |
| sub-graph | `grade` (per-criterion marks plus verbatim evidence quote) -> `verify` (code checks, then LLM critic) -> retry up to `MAX_GRADE_RETRIES` with the critique -> `finalize`. Unattempted questions skip the LLM. |
| `aggregate` | Totals are computed in code, never by the LLM. |
| `report` | Receives complete evaluations and review flags. Writes question-specific remarks, overall feedback and study priorities. Code renders awarded/maximum/withheld criterion marks, evidence and justifications; the LLM never computes totals. |

`auto_assessment/graph.py` assembles the top-level and per-question LangGraph workflows; `nodes.py` holds their callable node implementations, and `models.py` defines the Pydantic assessment models and dictionary-based graph state schemas. `schemas.py` and `state.py` retain compatibility imports.

**Grading safeguards (not a replacement for human review)**
- Every awarded criterion must quote the student verbatim, and the quote is fuzzy-matched against the answer in code.
- Marks are clamped to rubric bounds and summed in code.
- An LLM critic checks fairness (no credit for absent content, no penalty for valid alternatives) and feedback specificity.
- Questions that never pass verification are flagged in the report, not silently accepted.
- Uncertain handwriting is flagged for human review. Unattempted questions skip grading calls but remain in the report with zero marks and count towards the paper total.
- Extracted text and quotes retain their language/script. Feedback follows each question's language; report headings and overall remarks follow the predominant paper language. Accuracy depends on the selected text and vision models.
- Cleaning, evaluation, and report prompts live in [prompts.py](auto_assessment/prompts.py).

## LLM clients (auto_assessment/llm_client.py)
Separate `OllamaClient` and `OpenRouterClient` classes share the `LLMClient` interface for reference alignment, grading, verification, and reports:
`llm.structured(Schema, temperature=...)`.
`create_llm(settings)` selects the client using `LLM_PROVIDER`: `ollama` (default) or `openrouter`.
Clients retain LangChain structured-output validation and tracing. `invoke(prompt)` returns a LangChain message, including provider-supplied usage metadata; per-call temperatures do not mutate the shared model.
Existing `HybridLLM(settings)` calls remain supported through a compatibility alias in `llm.py`.
PDF/image extraction always uses OpenRouter, regardless of the grading provider. There is no local OCR fallback.

## Document extraction
- Set `OPENROUTER_API_KEY` for any PDF/image upload. Those files are sent to OpenRouter and its providers; OCR incurs API charges, including when grading uses Ollama.
- `OPENROUTER_VISION_MODEL` defaults to `mistralai/mistral-small-3.1-24b-instruct` and is separate from the grading text model. The UI exposes it as **OpenRouter OCR / vision model**.
- PDFs: send the entire original file using `type: file`, with `plugins: [{"id": "file-parser", "pdf": {"engine": "mistral-ocr"}}]`.
- Images: send original PNG/JPEG/WebP/GIF bytes using `image_url`. No PDF plugin or local image processing. BMP/TIFF uploads are no longer supported; convert them to a supported format before uploading.
- `.txt` / `.md` and pasted model answers need no OCR calls.
- Empty documents, HTTP/API errors, malformed replies, and incomplete OCR output stop assessment explicitly.
- Transcript boundaries use `=== DOCUMENT n ===`. State field `document_count` counts student files, not PDF pages; the pipeline no longer computes page counts locally.
- Uncertain `[illegible]` / `[?]` markers remain review warnings. Transcription accuracy still requires human review.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Set LLM_PROVIDER=ollama or LLM_PROVIDER=openrouter in .env.
# Set OPENROUTER_API_KEY for PDF/image OCR and for OpenRouter grading.
# OPENROUTER_VISION_MODEL=mistralai/mistral-small-3.1-24b-instruct
# Optionally add LANGSMITH_TRACING=true and LANGSMITH_API_KEY for tracing.
# for local models:
ollama pull qwen2.5:14b
```

## Check OpenRouter access and pricing
Set `OPENROUTER_API_KEY` in `.env`, then list models available to that key with
input/output prices per one million tokens:
```bash
python openrouter_models.py
python openrouter_models.py --search llama
python openrouter_models.py --model openai/gpt-4o --prompt "Say hello in one sentence."
```
The model list comes from OpenRouter's account-specific models endpoint. A model being
listed does not necessarily mean requests will succeed if the account has no remaining
credits or the model has other provider-specific requirements.

## Run
```bash
python app.py                                  # Web UI at http://127.0.0.1:8000

# CLI, no scans or rubric needed:
python -m auto_assessment.cli --question-paper examples/question_paper.txt \
    --model-answers examples/model_answers.txt --answers examples/student_answer.txt

# Run the example pipeline with node-by-node terminal logs:
python run_pipeline.py

# Optionally add explicit criterion allocations:
python -m auto_assessment.cli --question-paper examples/question_paper.txt \
    --model-answers examples/model_answers.txt --rubric examples/rubric.txt \
    --answers examples/student_answer.txt

# CLI with real scans:
python -m auto_assessment.cli --question-paper qp.pdf --rubric rubric.pdf \
    --model-answers model.pdf --answers page1.jpg page2.jpg page3.jpg

# Inline correct answers instead of a model-answer file:
python -m auto_assessment.cli --question-paper qp.pdf --rubric rubric.pdf \
    --model-answers-text "1. Correct answer ... 2. Correct answer ..." --answers student.pdf
```

CLI saves `report.md`, `result.json`, aligned `questions.json`, and `transcript.txt` to `out/`.
`python run_pipeline.py` runs the example inputs and saves outputs to `out/pipeline-test/`.
Pass CLI options to `run_pipeline.py` to run custom inputs. The graph logs each node's start,
completion time, output field names, question/attempt where applicable, and failures to the terminal;
student answers and model output are not included in logs.
UI accepts uploaded model answers or pasted correct answers. The **Grade & verify** step shows a
live per-question progress bar. Marks, criterion feedback, warnings, the transcript and JSON/Markdown
downloads live on the review sheet. After an assessment finishes, **Show reviews** in the sidebar
opens that review sheet (`/reviews`, above **How it works**) and stays disabled until a run
completes. **Clear & start new** removes the stored run and resets the studio.
The review sheet starts with total marks, percentage, attempt counts, human-review counts,
overall feedback, study priorities and assessment warnings. Search questions, student/model
answers, rubric criteria and feedback; filter by verification, unattempted answers or withheld
marks. Default order follows the question paper; optional sorts use natural question numbers,
human-review priority, withheld marks or score percentage. Jump links open question details,
with student/reference answers side by side, actionable feedback and criterion evidence.
Expand/collapse affects shown questions. **Print shown reviews** includes expanded details for
the current filtered list; **All scores CSV**, Markdown and JSON downloads always include the
complete result. CSV exports quote text and neutralize spreadsheet formulas.
The web UI uses Flask with plain HTML, CSS, and vanilla JavaScript; no Node.js, frontend build,
external fonts, or animation libraries are needed. It includes live streamed progress,
reorderable student pages and model settings; criterion feedback, review warnings and transcript
access live on the review sheet.
Animations respect reduced-motion preferences. Uploads are limited to 50 MB per request,
stored temporarily, and removed after processing. Results stay in the browser tab
(`sessionStorage`) across reloads and page switches until **Clear & start new** is used or the
tab is closed; download reports before leaving. Closing the page interrupts the progress
connection, not necessarily an in-flight model request. The server binds to loopback only; this unauthenticated
local tool is not intended for public hosting. Add authentication and a production server before
exposing it to a network.
Question paper, model answers (upload or pasted text), and at least one student sheet are mandatory.
Rubric is optional; when supplied, every criterion must have explicit marks.
Without a rubric, the system grades each question as a whole; it does not invent detailed marking allocations.
Structured JSON spec input is no longer supported.

## LangSmith
- Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`. Every run is traced end to end: node, per-question sub-graph, retries, LLM calls, with tags and metadata (provider, models).
- Evals: `python evals/run_eval.py` creates a dataset from `evals/dataset.json` and scores marks-in-bounds, total consistency, evidence grounding, verification rate, agreement with human marks, and LLM-judged feedback quality.
- Offline pipeline checks (no API needed): `python -m pytest tests/test_pipeline.py`
- Toy upload-to-review pipeline check: `python -m pytest tests/test_web.py -k toy_pipeline`
- Review sorting/search/export checks (optional Node.js, no packages): `node --test tests/test_reviews.js`

## Notes
- Extraction calls (`parse_spec`, `clean_align`, and PDF/image OCR) disable model reasoning by default for faster, cheaper, more literal output. Set `EXTRACTION_REASONING=true` to enable reasoning for those calls; grading, critic verification, and report generation are unaffected.
- Handwriting accuracy is the main risk. Ollama grading is fully local only with text inputs; PDF/image OCR always uses OpenRouter.
- Diagrams, graphs and tables in images and scanned PDFs are converted into detailed structured text during OCR and graded from that text. For PDFs, `mistral-ocr` forwards up to 8 extracted images per PDF to the vision model; any beyond that limit may only appear as parsed text. There is still no per-question image crop or visual re-check during grading.
- All document roles accept text/scanned PDFs, `.txt`, `.md`, or PNG/JPEG/WebP/GIF images. Reference correct answers may also be pasted as text.
- Ollama context defaults are small, so `OLLAMA_NUM_CTX=8192` is set.
