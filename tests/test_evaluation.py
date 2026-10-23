"""
Tests for the production retrieval evaluator.

These tests use a fake retrieve_fn so they do not need sentence-transformers.
The verified QA JSON is loaded from disk to lock the dataset schema.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    EvalCase,
    EvaluationError,
    evaluate_cases,
    load_verified_qa_set,
)


QA_PATH = PROJECT_ROOT / "data" / "sample_materials" / "verified_qa_set.json"


def _retrieval(chunks, has_evidence, all_results=None):
    return {
        "chunks": chunks,
        "has_evidence": has_evidence,
        "all_results": all_results if all_results is not None else chunks,
        "evidence_count": len(chunks),
    }


def test_load_verified_qa_set_schema():
    cases = load_verified_qa_set(QA_PATH)
    assert len(cases) == 9
    in_scope = [c for c in cases if c.in_scope]
    out_of_scope = [c for c in cases if not c.in_scope]
    assert len(in_scope) == 8
    assert len(out_of_scope) == 1
    assert out_of_scope[0].id == "unsupported_001"
    assert out_of_scope[0].expected_behavior == "insufficient_evidence"


def test_load_verified_qa_set_missing_file(tmp_path):
    with pytest.raises(EvaluationError, match="not found"):
        load_verified_qa_set(tmp_path / "missing.json")


def test_load_verified_qa_set_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "qa.json"
    path.write_text(
        json.dumps(
            [
                {"id": "a", "question": "Q1", "expected_source": "notes.txt"},
                {"id": "a", "question": "Q2", "expected_source": "notes.txt"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="Duplicate"):
        load_verified_qa_set(path)


def test_load_verified_qa_set_rejects_empty_question(tmp_path):
    path = tmp_path / "qa.json"
    path.write_text(
        json.dumps([{"id": "a", "question": "  ", "expected_source": "notes.txt"}]),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="question"):
        load_verified_qa_set(path)


def test_evaluate_cases_hit_and_out_of_scope():
    cases = [
        EvalCase(
            id="in_1",
            question="What is overfitting?",
            expected_keywords=("overfitting", "bias"),
            expected_source="ml.txt",
        ),
        EvalCase(
            id="out_1",
            question="Capital of France?",
            expected_behavior="insufficient_evidence",
        ),
    ]

    def retrieve_fn(query: str) -> dict:
        if "overfitting" in query.lower():
            return _retrieval(
                [
                    {
                        "text": "Overfitting is high variance; bias causes underfitting.",
                        "source": "ml.txt",
                        "score": 0.8,
                    }
                ],
                has_evidence=True,
            )
        return _retrieval([], has_evidence=False, all_results=[{"source": "ml.txt", "score": 0.1}])

    report = evaluate_cases(cases, retrieve_fn, known_sources={"ml.txt"})
    assert report.n_in_scope == 1
    assert report.n_out_of_scope == 1
    assert report.retrieval_hit_rate == 1.0
    assert report.evidence_threshold_accuracy_in_scope == 1.0
    assert report.insufficient_evidence_accuracy == 1.0
    assert report.citation_accuracy == 1.0
    assert report.mean_keyword_recall == 1.0
    assert report.cases[0].retrieval_hit is True
    assert report.cases[1].evidence_correct is True


def test_evaluate_cases_missed_source_and_false_positive():
    cases = [
        EvalCase(id="in_1", question="p-value?", expected_source="stats.txt"),
        EvalCase(
            id="out_1",
            question="France?",
            expected_behavior="insufficient_evidence",
        ),
    ]

    def retrieve_fn(query: str) -> dict:
        return _retrieval(
            [{"text": "unrelated", "source": "ml.txt", "score": 0.4}],
            has_evidence=True,
        )

    report = evaluate_cases(cases, retrieve_fn, known_sources={"ml.txt", "stats.txt"})
    assert report.retrieval_hit_rate == 0.0
    assert report.insufficient_evidence_accuracy == 0.0
    assert report.cases[0].notes.startswith("missed")
    assert "false positive" in report.cases[1].notes


def test_evaluate_cases_rejects_unknown_citation():
    cases = [EvalCase(id="in_1", question="overfitting?", expected_source="ml.txt")]

    def retrieve_fn(_query: str) -> dict:
        return _retrieval(
            [{"text": "Overfitting", "source": "invented.pdf", "score": 0.9}],
            has_evidence=True,
        )

    report = evaluate_cases(cases, retrieve_fn, known_sources={"ml.txt"})
    assert report.citation_accuracy == 0.0
    assert report.cases[0].citations_valid is False


def test_markdown_report_contains_table():
    cases = [EvalCase(id="in_1", question="overfitting?", expected_source="ml.txt")]

    def retrieve_fn(_query: str) -> dict:
        return _retrieval(
            [{"text": "Overfitting", "source": "ml.txt", "score": 0.9}],
            has_evidence=True,
        )

    report = evaluate_cases(cases, retrieve_fn, known_sources={"ml.txt"})
    md = report.to_markdown()
    assert "| in_1 |" in md
    assert "Retrieval hit rate" in md


def test_evaluate_cases_empty_raises():
    with pytest.raises(EvaluationError, match="No evaluation cases"):
        evaluate_cases([], lambda q: _retrieval([], False))
