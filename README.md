# Auto-Assessment: agentic grading of handwritten answer sheets

LangGraph (orchestration) + LangChain (models, structured output, fallbacks) + LangSmith (tracing and evals).

**Input:** question paper, handwritten answer-sheet scans, rubric, model answer per question
**Output:** marks per question + rubric-defensible, student-specific feedback (Markdown report + JSON)

## Architecture

```
            +--> parse_spec --+
   START ---+                 +--> segment --Send x N--> grade_question --> aggregate --> summarize --> END
            +--> transcribe --+                          |
                                                         v   (sub-graph, one per question)
                                           grade -> verify -> [issues? retry with critique] -> finalize
```

| Node | What it does |
|---|---|
| `parse_spec` | LLM turns the question paper, rubric and model answers into a typed `ExamSpec`. Criteria marks are rescaled in code if they do not sum to the question total. Skipped if you pass a JSON spec. |
| `transcribe` | Vision model reads each page with a "transcribe, do not correct" prompt. Pages run in parallel. |
| `segment` | Maps the transcript to question ids (answers can span pages). Flags segments that do not match the transcript. |
| `grade_question` | Fans out with `Send`, one sub-graph per question. |
| sub-graph | `grade` (per-criterion marks plus verbatim evidence quote) -> `verify` (code checks, then LLM critic) -> retry up to `MAX_GRADE_RETRIES` with the critique -> `finalize`. Unattempted questions skip the LLM. |
| `aggregate` | Totals are computed in code, never by the LLM. |
| `summarize` | Overall feedback and study priorities, then the Markdown report. |

**Why it is defensible without a human reviewer**
- Every awarded criterion must quote the student verbatim, and the quote is fuzzy-matched against the answer in code.
- Marks are clamped to rubric bounds and summed in code.
- An LLM critic checks fairness (no credit for absent content, no penalty for valid alternatives) and feedback specificity.
- Questions that never pass verification are flagged in the report, not silently accepted.

## `HybridLLM` (auto_assessment/llm.py)
One class wrapping OpenAI and Ollama. Every node calls `llm.structured(Schema, vision=...)`.
`LLM_PROVIDER` = `openai` | `ollama` | `openai_then_ollama` | `ollama_then_openai`
(fallback uses LangChain `.with_fallbacks`; OpenAI is skipped automatically if no API key is set).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add OPENAI_API_KEY, LANGSMITH_API_KEY
# for local models:
ollama pull qwen2.5:14b && ollama pull qwen2.5vl:7b
```

## Run
```bash
streamlit run app.py                            # Streamlit UI

# CLI, no scans needed (typed .txt answer + JSON spec):
python -m auto_assessment.cli --spec examples/spec.json --answers examples/student_answer.txt

# CLI with real scans:
python -m auto_assessment.cli --question-paper qp.pdf --rubric rubric.pdf \
    --model-answers model.pdf --answers page1.jpg page2.jpg page3.jpg
```

## LangSmith
- Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`. Every run is traced end to end: node, per-question sub-graph, retries, LLM calls, with tags and metadata (provider, models).
- Evals: `python evals/run_eval.py` creates a dataset from `evals/dataset.json` and scores marks-in-bounds, total consistency, evidence grounding, verification rate, agreement with human marks, and LLM-judged feedback quality.
- Tests (no API needed): `pytest tests/`

## Notes
- Handwriting accuracy is the main risk. GPT-4o-class vision models are far better than small local VLMs; use `ollama` mode mainly for privacy or offline demos.
- Question paper, rubric and model answers must be text PDFs, `.txt`, `.md` or JSON. Scanned versions of those need a transcription step first.
- Ollama context defaults are small, so `OLLAMA_NUM_CTX=8192` is set.
