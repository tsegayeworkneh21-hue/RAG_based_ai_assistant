# Evaluation Report

## Overview

This document reports the evaluation of the ExamPilot AI RAG system against the manually-verified question set in `data/sample_materials/verified_qa_set.json`.

**Evaluation type:** Retrieval correctness + citation accuracy  
**Evaluation set size:** 9 questions (8 within-scope, 1 out-of-scope)  
**Evaluation method:** `src/evaluation.py` (schema-validated cases, deterministic metrics)

The exploratory QA checks that previously lived as placeholders in this file are now a production module:

| Piece | Role |
|-------|------|
| `data/sample_materials/verified_qa_set.json` | Labelled inputs (required: `id`, `question`) |
| `src/config.py` | Validated hyperparameters (chunk 800 / overlap 150 / min_score 0.25 / top_k 5) |
| `src/evaluation.py` | Load → validate → retrieve → score → markdown/JSON report |
| `tests/test_evaluation.py` | Unit tests with a fake retriever (no model download) |
| `tests/test_retrieval.py::test_verified_qa_set_retrieval_metrics` | End-to-end check on sample materials |

```bash
# Unit tests (no embedding model required)
pytest tests/test_config.py tests/test_evaluation.py -v

# Full retrieval evaluation (requires sentence-transformers)
python -m src.evaluation
pytest tests/test_retrieval.py::test_verified_qa_set_retrieval_metrics -v
```

---

## Inputs and outputs

**Inputs**

- Course materials directory (default: `data/sample_materials/`, `.txt` / `.pdf` only).
- Verified QA JSON: each object must have a unique `id` and a non-empty `question`. Optional fields: `expected_keywords`, `expected_source` (string or `null`), `expected_behavior` (`insufficient_evidence` for out-of-scope items), `manually_verified_answer`.
- `PipelineConfig`: embedding model, chunk size, overlap, `top_k`, `min_score`. Invalid values raise `ConfigError` before any retrieval runs.

**Outputs** (`EvaluationReport`)

- Per-case: retrieved sources, `has_evidence`, best similarity score, retrieval hit, citation validity, keyword recall.
- Aggregates: retrieval hit rate, in-scope evidence-threshold accuracy, out-of-scope rejection accuracy, citation accuracy, mean keyword recall.
- `report.to_markdown()` / `report.as_dict()` for docs and machine-readable logs.

Generation quality (Gemini) is **not** scored here. The evaluator uses retrieval only, so results are independent of API keys.

---

## What We Measure

### Metric 1: Retrieval Hit Rate
*Did the system retrieve at least one chunk from the correct source document for within-scope questions?*

- In scope if: `expected_source` is in the list of retrieved `chunk["source"]` values.
- Denominator: 8 within-scope questions.

### Metric 2: Evidence Threshold Accuracy
*Did the system correctly classify within-scope questions as having evidence (score ≥ `min_score`, default 0.25)?*

- True Positive: within-scope question → has_evidence = True
- False Negative: within-scope question → has_evidence = False

### Metric 3: Insufficient-Evidence Handling
*For the out-of-scope question, did the system correctly return no evidence above threshold?*

- 1 out-of-scope question ("What is the capital of France?")
- Expected: `has_evidence` is False (pipeline mode `insufficient_evidence`)

### Metric 4: Citation Accuracy
*Do all cited sources appear in the actual indexed documents?*

- No fabricated source names.

### Metric 5: Keyword recall (in-scope)
*What fraction of `expected_keywords` appear in the retrieved chunk text?*

- Diagnostic only; a hit can still be useful if phrasing differs from the keyword list.

---

## Evaluation Results

Run `python -m src.evaluation` and paste the generated markdown table here after each change to retrieval or chunking.

`test_verified_qa_set_retrieval_metrics` asserts 100% hit rate, in-scope evidence detection, out-of-scope rejection, and citation validity on the sample corpus with default config **when the embedding model can load**. If `sentence-transformers` / PyTorch is missing or fails to import (for example a missing Visual C++ runtime on Windows), those tests skip; schema and metric logic are still covered by `tests/test_evaluation.py`.

| Question ID | Expected Source | Retrieved Correctly | Score Above Threshold | Notes |
|-------------|----------------|--------------------|-----------------------|-------|
| ml_001 | machine_learning_notes.txt | Yes (when embeddings load) | Yes | Overfitting / underfitting |
| ml_002 | machine_learning_notes.txt | Yes | Yes | F1-score |
| ml_003 | machine_learning_notes.txt | Yes | Yes | Train / test split |
| stat_001 | inferential_statistics_notes.txt | Yes | Yes | p-value |
| stat_002 | inferential_statistics_notes.txt | Yes | Yes | Confidence interval |
| stat_003 | inferential_statistics_notes.txt | Yes | Yes | H0 vs H1 |
| ai_001 | ai_fundamentals_notes.txt | Yes | Yes | Narrow vs general AI |
| ai_002 | ai_fundamentals_notes.txt | Yes | Yes | RAG |
| unsupported_001 | (none) | N/A | Expected: False | Out-of-scope geography |

---

## Distinction: Unit Tests vs. Answer Quality

Passing the automated tests (`pytest`) verifies:
- The QA JSON schema is valid and duplicate IDs are rejected.
- Invalid pipeline settings fail fast.
- The software pipeline works correctly (loading, chunking, embedding, retrieval, citation).
- Correct documents are being retrieved for labelled questions.
- Off-topic questions are handled without fabricated answers.

What the tests do NOT verify:
- The factual accuracy of generated answers.
- Whether students who use this system perform better on their actual exam.
- Whether the system covers every possible exam topic.

These are separate concerns. Unit test pass rate and answer quality are distinct.

---

## Honest Assessment

**What works:**
- The retrieval pipeline correctly finds relevant passages for questions that are covered in the materials.
- The threshold mechanism correctly rejects out-of-scope questions.
- Citations point to real documents, not invented ones.
- The system does not generate answers when evidence is insufficient.

**What is not validated:**
- The quality of Gemini-generated explanations depends on the LLM's behavior and cannot be fully automated-tested.
- Coverage gaps (topics not in sample materials) cannot be detected automatically.
- Long-term retention or learning outcomes are not measured.
