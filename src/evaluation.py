"""
evaluation.py
=============
Production evaluation of retrieval against the verified QA set.

This module turns the exploratory checks in docs/EVALUATION.md and
data/sample_materials/verified_qa_set.json into a repeatable function:

    load cases → validate schema → retrieve → score → report

INPUTS
------
- A JSON list of evaluation cases (see load_verified_qa_set).
- A retrieve_fn(query) -> retrieval dict, or a built VectorIndex.
- PipelineConfig (threshold, top_k).

OUTPUTS
-------
EvaluationReport with:
- per-case results (hit, evidence decision, citation validity, keyword recall)
- aggregate metrics matching docs/EVALUATION.md
- a markdown table suitable for the evaluation report

The evaluator scores **retrieval**, not LLM answer quality. Generation quality
depends on the model and is documented separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Allow running this file directly as `python src/evaluation.py` from the repo root
# without requiring an installed package context.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config import DEFAULT_CONFIG, PipelineConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QA_PATH = PROJECT_ROOT / "data" / "sample_materials" / "verified_qa_set.json"
DEFAULT_MATERIALS_DIR = PROJECT_ROOT / "data" / "sample_materials"

RetrieveFn = Callable[[str], dict]


class EvaluationError(ValueError):
    """Raised when the evaluation dataset is missing or malformed."""


@dataclass(frozen=True)
class EvalCase:
    """One labelled question from the verified QA set."""

    id: str
    question: str
    expected_keywords: tuple[str, ...] = ()
    expected_source: Optional[str] = None
    expected_behavior: Optional[str] = None
    manually_verified_answer: Optional[str] = None

    @property
    def in_scope(self) -> bool:
        """True when the materials are expected to contain an answer."""
        if self.expected_behavior == "insufficient_evidence":
            return False
        return self.expected_source is not None


@dataclass
class CaseResult:
    """Scored outcome for a single evaluation case."""

    case_id: str
    question: str
    in_scope: bool
    expected_source: Optional[str]
    retrieved_sources: list[str]
    has_evidence: bool
    best_score: float
    retrieval_hit: Optional[bool]
    evidence_correct: bool
    citations_valid: bool
    keyword_recall: Optional[float]
    notes: str


@dataclass
class EvaluationReport:
    """Aggregate retrieval metrics plus per-case detail."""

    config: dict
    n_cases: int
    n_in_scope: int
    n_out_of_scope: int
    retrieval_hit_rate: Optional[float]
    evidence_threshold_accuracy_in_scope: Optional[float]
    insufficient_evidence_accuracy: Optional[float]
    citation_accuracy: Optional[float]
    mean_keyword_recall: Optional[float]
    cases: list[CaseResult] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "config": self.config,
            "n_cases": self.n_cases,
            "n_in_scope": self.n_in_scope,
            "n_out_of_scope": self.n_out_of_scope,
            "retrieval_hit_rate": self.retrieval_hit_rate,
            "evidence_threshold_accuracy_in_scope": self.evidence_threshold_accuracy_in_scope,
            "insufficient_evidence_accuracy": self.insufficient_evidence_accuracy,
            "citation_accuracy": self.citation_accuracy,
            "mean_keyword_recall": self.mean_keyword_recall,
            "cases": [case_result_to_dict(c) for c in self.cases],
        }

    def to_markdown(self) -> str:
        """Render a markdown table of per-case results plus summary metrics."""
        lines = [
            "## Automated retrieval evaluation",
            "",
            f"- Cases: {self.n_cases} ({self.n_in_scope} in-scope, {self.n_out_of_scope} out-of-scope)",
            f"- Retrieval hit rate: {_fmt_pct(self.retrieval_hit_rate)}",
            f"- In-scope evidence threshold accuracy: {_fmt_pct(self.evidence_threshold_accuracy_in_scope)}",
            f"- Out-of-scope insufficient-evidence accuracy: {_fmt_pct(self.insufficient_evidence_accuracy)}",
            f"- Citation accuracy: {_fmt_pct(self.citation_accuracy)}",
            f"- Mean keyword recall (in-scope): {_fmt_pct(self.mean_keyword_recall)}",
            "",
            "| Question ID | Expected Source | Retrieved Correctly | Score Above Threshold | Notes |",
            "|-------------|----------------|--------------------|-----------------------|-------|",
        ]
        for case in self.cases:
            expected = case.expected_source or "(none)"
            if case.in_scope:
                hit = _bool_cell(case.retrieval_hit)
                above = _bool_cell(case.has_evidence)
            else:
                hit = "N/A"
                above = "Expected: False" if not case.has_evidence else "False (leaked evidence)"
            lines.append(
                f"| {case.case_id} | {expected} | {hit} | {above} | {case.notes} |"
            )
        return "\n".join(lines) + "\n"


def case_result_to_dict(result: CaseResult) -> dict:
    return {
        "case_id": result.case_id,
        "question": result.question,
        "in_scope": result.in_scope,
        "expected_source": result.expected_source,
        "retrieved_sources": result.retrieved_sources,
        "has_evidence": result.has_evidence,
        "best_score": result.best_score,
        "retrieval_hit": result.retrieval_hit,
        "evidence_correct": result.evidence_correct,
        "citations_valid": result.citations_valid,
        "keyword_recall": result.keyword_recall,
        "notes": result.notes,
    }


def load_verified_qa_set(path: str | Path) -> list[EvalCase]:
    """
    Load and validate the verified QA JSON file.

    Expected schema (list of objects)
    ---------------------------------
    id                      : non-empty string (required)
    question                : non-empty string (required)
    expected_keywords       : list of strings (optional, default [])
    expected_source         : filename string or null (optional)
    expected_behavior       : "insufficient_evidence" or omitted
    manually_verified_answer: string (optional, documentation only)

    Raises
    ------
    EvaluationError
        If the file is missing, not JSON, or any case fails validation.
    """
    qa_path = Path(path)
    if not qa_path.is_file():
        raise EvaluationError(f"QA set not found: {qa_path}")

    try:
        raw = json.loads(qa_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"QA set is not valid JSON: {qa_path}") from exc

    if not isinstance(raw, list) or not raw:
        raise EvaluationError("QA set must be a non-empty JSON list.")

    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(raw):
        cases.append(_parse_case(item, index=i, seen_ids=seen_ids))
    return cases


def evaluate_cases(
    cases: list[EvalCase],
    retrieve_fn: RetrieveFn,
    known_sources: Optional[set[str]] = None,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> EvaluationReport:
    """
    Score every case using a retrieve function.

    Parameters
    ----------
    cases         : Validated EvalCase list.
    retrieve_fn   : Callable mapping a question string to a retrieve() dict.
                    Must include keys: chunks, has_evidence, all_results.
    known_sources : Source filenames present in the index. If provided, every
                    retrieved citation must belong to this set.
    config        : Pipeline settings recorded on the report (not applied here;
                    the retrieve_fn is already configured).

    Returns
    -------
    EvaluationReport
    """
    if not cases:
        raise EvaluationError("No evaluation cases provided.")

    results = [
        _score_case(case, retrieve_fn(case.question), known_sources)
        for case in cases
    ]
    return _aggregate(results, config)


def run_retrieval_evaluation(
    materials_dir: str | Path = DEFAULT_MATERIALS_DIR,
    qa_path: str | Path = DEFAULT_QA_PATH,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> EvaluationReport:
    """
    Build an index from course materials and evaluate the verified QA set.

    This is the end-to-end production entry point. It requires
    sentence-transformers to be installed.
    """
    from src.document_processing import load_all_from_directory
    from src.rag_pipeline import build_knowledge_base
    from src.retrieval import retrieve

    materials = Path(materials_dir)
    chunks = load_all_from_directory(
        str(materials),
        chunk_size=config.chunk_size,
        overlap=config.overlap,
    )
    if not chunks:
        raise EvaluationError(f"No chunks loaded from {materials}")

    index = build_knowledge_base(chunks, model_name=config.embedding_model)
    if not index.is_built:
        raise EvaluationError("Failed to build the vector index.")

    known_sources = {chunk["source"] for chunk in chunks}
    cases = load_verified_qa_set(qa_path)

    def retrieve_fn(query: str) -> dict:
        return retrieve(query, index, top_k=config.top_k, min_score=config.min_score)

    return evaluate_cases(cases, retrieve_fn, known_sources=known_sources, config=config)


def _parse_case(item: object, index: int, seen_ids: set[str]) -> EvalCase:
    if not isinstance(item, dict):
        raise EvaluationError(f"Case {index} must be a JSON object.")

    case_id = item.get("id")
    question = item.get("question")
    if not isinstance(case_id, str) or not case_id.strip():
        raise EvaluationError(f"Case {index} is missing a non-empty 'id'.")
    if case_id in seen_ids:
        raise EvaluationError(f"Duplicate case id: {case_id}")
    seen_ids.add(case_id)
    if not isinstance(question, str) or not question.strip():
        raise EvaluationError(f"Case {case_id} is missing a non-empty 'question'.")

    keywords = item.get("expected_keywords", [])
    if keywords is None:
        keywords = []
    if not isinstance(keywords, list) or any(not isinstance(k, str) for k in keywords):
        raise EvaluationError(f"Case {case_id}: expected_keywords must be a list of strings.")

    expected_source = item.get("expected_source")
    if expected_source is not None and not isinstance(expected_source, str):
        raise EvaluationError(f"Case {case_id}: expected_source must be a string or null.")

    expected_behavior = item.get("expected_behavior")
    if expected_behavior is not None and not isinstance(expected_behavior, str):
        raise EvaluationError(f"Case {case_id}: expected_behavior must be a string or null.")

    answer = item.get("manually_verified_answer")
    if answer is not None and not isinstance(answer, str):
        raise EvaluationError(f"Case {case_id}: manually_verified_answer must be a string or null.")

    return EvalCase(
        id=case_id.strip(),
        question=question.strip(),
        expected_keywords=tuple(keywords),
        expected_source=expected_source,
        expected_behavior=expected_behavior,
        manually_verified_answer=answer,
    )


def _score_case(
    case: EvalCase,
    retrieval: dict,
    known_sources: Optional[set[str]],
) -> CaseResult:
    if not isinstance(retrieval, dict):
        raise EvaluationError(f"retrieve_fn must return a dict for case {case.id}.")

    chunks = retrieval.get("chunks") or []
    all_results = retrieval.get("all_results") or []
    has_evidence = bool(retrieval.get("has_evidence"))
    retrieved_sources = [str(c.get("source", "")) for c in chunks]

    best_score = 0.0
    score_pool = chunks or all_results
    if score_pool:
        best_score = float(score_pool[0].get("score") or 0.0)

    if case.in_scope:
        retrieval_hit = case.expected_source in retrieved_sources
        evidence_correct = has_evidence
        notes = "hit" if retrieval_hit else "missed expected source"
        if not has_evidence:
            notes = "false negative (no evidence above threshold)"
    else:
        retrieval_hit = None
        evidence_correct = not has_evidence
        notes = "correctly rejected" if evidence_correct else "false positive (out-of-scope evidence)"

    if known_sources is None:
        citations_valid = True
    else:
        citations_valid = all(src in known_sources for src in retrieved_sources)

    if not citations_valid:
        notes = "fabricated or unknown citation source"

    keyword_recall = None
    if case.in_scope and case.expected_keywords:
        blob = " ".join(c.get("text", "") for c in chunks).lower()
        hits = sum(1 for kw in case.expected_keywords if kw.lower() in blob)
        keyword_recall = hits / len(case.expected_keywords)

    return CaseResult(
        case_id=case.id,
        question=case.question,
        in_scope=case.in_scope,
        expected_source=case.expected_source,
        retrieved_sources=retrieved_sources,
        has_evidence=has_evidence,
        best_score=best_score,
        retrieval_hit=retrieval_hit,
        evidence_correct=evidence_correct,
        citations_valid=citations_valid,
        keyword_recall=keyword_recall,
        notes=notes,
    )


def _aggregate(results: list[CaseResult], config: PipelineConfig) -> EvaluationReport:
    in_scope = [r for r in results if r.in_scope]
    out_of_scope = [r for r in results if not r.in_scope]

    hit_rate = None
    if in_scope:
        hit_rate = sum(1 for r in in_scope if r.retrieval_hit) / len(in_scope)

    in_scope_threshold = None
    if in_scope:
        in_scope_threshold = sum(1 for r in in_scope if r.evidence_correct) / len(in_scope)

    oos_accuracy = None
    if out_of_scope:
        oos_accuracy = sum(1 for r in out_of_scope if r.evidence_correct) / len(out_of_scope)

    cited = [r for r in results if r.retrieved_sources]
    citation_accuracy = None
    if cited:
        citation_accuracy = sum(1 for r in cited if r.citations_valid) / len(cited)

    keyword_vals = [r.keyword_recall for r in in_scope if r.keyword_recall is not None]
    mean_kw = sum(keyword_vals) / len(keyword_vals) if keyword_vals else None

    return EvaluationReport(
        config=config.as_dict(),
        n_cases=len(results),
        n_in_scope=len(in_scope),
        n_out_of_scope=len(out_of_scope),
        retrieval_hit_rate=hit_rate,
        evidence_threshold_accuracy_in_scope=in_scope_threshold,
        insufficient_evidence_accuracy=oos_accuracy,
        citation_accuracy=citation_accuracy,
        mean_keyword_recall=mean_kw,
        cases=results,
    )


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _bool_cell(value: Optional[bool]) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "—"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ExamPilot retrieval against the verified QA set."
    )
    parser.add_argument(
        "--materials",
        type=Path,
        default=DEFAULT_MATERIALS_DIR,
        help="Directory of .txt/.pdf course materials.",
    )
    parser.add_argument(
        "--qa-set",
        type=Path,
        default=DEFAULT_QA_PATH,
        help="Path to verified_qa_set.json.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_CONFIG.min_score,
        help="Cosine similarity threshold (default: 0.25).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_CONFIG.top_k,
        help="Number of chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON instead of markdown.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    config = PipelineConfig(min_score=args.min_score, top_k=args.top_k)
    report = run_retrieval_evaluation(
        materials_dir=args.materials,
        qa_path=args.qa_set,
        config=config,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(report.to_markdown())
        failed = [
            c for c in report.cases
            if not c.evidence_correct or not c.citations_valid or c.retrieval_hit is False
        ]
        if failed:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
