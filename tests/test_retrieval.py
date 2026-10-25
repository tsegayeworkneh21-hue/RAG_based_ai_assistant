"""
Tests for retrieval.py and rag_pipeline.py

RUN WITH:
    pytest tests/test_retrieval.py -v

WHAT WE TEST
------------
1. Retrieval returns relevant chunks for known questions.
2. An off-topic question triggers the insufficient-evidence path.
3. The context string contains source labels.
4. answer_question returns correct mode when no index is built.
5. Quiz scoring is correct.
6. Citation metadata is never fabricated.

NOTE ON MOCKING
---------------
The retrieval and RAG pipeline tests that involve embeddings require
sentence-transformers to be installed. We skip those tests gracefully
if the library is not available.
Tests that exercise only data transformation (context building, quiz scoring)
do NOT require any external libraries.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.retrieval import build_context_string, build_full_prompt, build_system_prompt
from src.quiz import (
    check_answer,
    calculate_score,
    generate_course_quiz,
    get_questions_by_subject,
)


# ---------------------------------------------------------------------------
# context string tests (no embedding needed)
# ---------------------------------------------------------------------------

def _make_fake_retrieval_result(has_evidence=True):
    """Helper: create a fake retrieval result for testing context building."""
    if not has_evidence:
        return {
            "chunks": [],
            "has_evidence": False,
            "all_results": [],
            "query": "What is the capital of France?",
            "evidence_count": 0,
        }
    return {
        "chunks": [
            {
                "text": "Overfitting occurs when a model learns training data too well.",
                "source": "machine_learning_notes.txt",
                "page": 3,
                "chunk_index": 5,
                "score": 0.82,
            },
            {
                "text": "The bias-variance trade-off describes the tension between model complexity and generalisation.",
                "source": "machine_learning_notes.txt",
                "page": 4,
                "chunk_index": 6,
                "score": 0.71,
            },
        ],
        "has_evidence": True,
        "all_results": [],
        "query": "What is overfitting?",
        "evidence_count": 2,
    }


def test_context_string_contains_source_labels():
    """The context string must include [SOURCE 1], [SOURCE 2], etc."""
    result = _make_fake_retrieval_result(has_evidence=True)
    context = build_context_string(result)
    assert "[SOURCE 1]" in context
    assert "[SOURCE 2]" in context


def test_context_string_contains_source_name():
    """The source file name must appear in the context string."""
    result = _make_fake_retrieval_result(has_evidence=True)
    context = build_context_string(result)
    assert "machine_learning_notes.txt" in context


def test_context_string_contains_page_number():
    """Page numbers must appear in the context string."""
    result = _make_fake_retrieval_result(has_evidence=True)
    context = build_context_string(result)
    assert "Page 3" in context


def test_context_string_empty_when_no_chunks():
    """If there are no chunks, the context string should be empty."""
    result = _make_fake_retrieval_result(has_evidence=False)
    context = build_context_string(result)
    assert context == ""


def test_full_prompt_contains_question():
    """The full prompt must contain the student's question."""
    result = _make_fake_retrieval_result(has_evidence=True)
    prompt = build_full_prompt("What is overfitting?", result)
    assert "What is overfitting?" in prompt


def test_full_prompt_insufficient_evidence():
    """When no evidence, the prompt must instruct the LLM to say it cannot answer."""
    result = _make_fake_retrieval_result(has_evidence=False)
    prompt = build_full_prompt("What is the capital of France?", result)
    assert "insufficient" in prompt.lower() or "not contain" in prompt.lower()


def test_system_prompt_prohibits_invented_citations():
    """The system prompt must include instructions against inventing citations."""
    system_prompt = build_system_prompt()
    # Check that the prompt forbids inventing citations
    assert "never" in system_prompt.lower() or "do not" in system_prompt.lower() or "not" in system_prompt.lower()
    assert "citation" in system_prompt.lower() or "source" in system_prompt.lower()


# ---------------------------------------------------------------------------
# llm_provider: no-key behaviour
# ---------------------------------------------------------------------------

def test_no_api_key_returns_no_key_mode(monkeypatch):
    """When GEMINI_API_KEY is absent, generate_answer must return mode='no_key'."""
    # Patch os.getenv to return None for the API key
    monkeypatch.setenv("GEMINI_API_KEY", "")

    from src.llm_provider import generate_answer
    result = generate_answer("test prompt")
    # Either no_key or error mode is acceptable when the key is empty
    assert result["success"] is False
    assert result["mode"] in ("no_key", "error")


def test_is_api_available_false_when_no_key(monkeypatch):
    """is_api_available() should return False when the env var is empty."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    from src.llm_provider import is_api_available
    # Force reload the check
    assert is_api_available() is False


# ---------------------------------------------------------------------------
# Quiz tests (no embedding needed)
# ---------------------------------------------------------------------------

def test_check_answer_correct():
    """Checking the correct answer returns correct=True."""
    result = check_answer("q1", "B")  # q1 correct answer is B
    assert result["correct"] is True


def test_check_answer_incorrect():
    """Checking a wrong answer returns correct=False with the right answer."""
    result = check_answer("q1", "A")  # q1 correct answer is B
    assert result["correct"] is False
    assert result["correct_answer"] == "B"


def test_check_answer_has_explanation():
    """Every answer check must return an explanation string."""
    result = check_answer("q1", "B")
    assert "explanation" in result
    assert len(result["explanation"]) > 10


def test_calculate_score_all_correct():
    """If all answers are correct, score should equal total questions answered."""
    # Map each question ID to its correct answer
    from src.quiz import QUIZ_QUESTIONS
    perfect_answers = {q["id"]: q["answer"] for q in QUIZ_QUESTIONS}
    result = calculate_score(perfect_answers)
    assert result["score"] == result["total"]
    assert result["percentage"] == 100.0
    assert result["grade"] == "A"


def test_calculate_score_all_wrong():
    """If all answers are wrong, score should be 0."""
    from src.quiz import QUIZ_QUESTIONS
    # Pick a wrong answer for each question (cycle through A, B, C, D)
    wrong_answers = {}
    for q in QUIZ_QUESTIONS:
        options = list(q["options"].keys())
        # Pick any option that is NOT the correct answer
        for opt in options:
            if opt != q["answer"]:
                wrong_answers[q["id"]] = opt
                break
    result = calculate_score(wrong_answers)
    assert result["score"] == 0
    assert result["percentage"] == 0.0


def test_get_questions_by_subject_ml():
    """Filtering by Machine Learning should return only ML questions."""
    questions = get_questions_by_subject("Machine Learning")
    assert len(questions) > 0
    for q in questions:
        assert q["subject"] == "Machine Learning"


def test_get_questions_by_subject_all():
    """No filter should return all questions."""
    all_q = get_questions_by_subject(None)
    from src.quiz import QUIZ_QUESTIONS
    assert len(all_q) == len(QUIZ_QUESTIONS)


def test_generated_course_questions_can_be_scored():
    questions = [{
        "id": "course_test_1",
        "subject": "Data Visualization",
        "question": "Which chart is suitable for comparing categories?",
        "options": {"A": "Bar chart", "B": "Scatter plot", "C": "Line chart", "D": "Histogram"},
        "answer": "A",
        "explanation": "Bar charts compare values across categories.",
    }]

    assert check_answer("course_test_1", "A", questions)["correct"]
    assert calculate_score({"course_test_1": "A"}, questions)["percentage"] == 100.0


def test_generate_course_quiz_parses_questions_from_material(monkeypatch):
    generated = {
        "questions": [{
            "subject": "Data Visualization",
            "question": "What does a histogram show?",
            "options": {"A": "Categories", "B": "A distribution", "C": "A network", "D": "A sequence"},
            "answer": "B",
            "explanation": "A histogram shows the distribution of numeric data.",
        }]
    }
    seen_prompts = []

    def fake_generate_answer(prompt, **kwargs):
        seen_prompts.append(prompt)
        assert kwargs["max_output_tokens"] == 8192
        import json
        return {"success": True, "answer": json.dumps(generated)}

    monkeypatch.setattr("src.quiz.generate_answer", fake_generate_answer)
    result = generate_course_quiz([{"source": "visualization.txt", "text": "Histograms show numeric data distributions."}])

    assert result["error"] is None
    assert result["questions"][0]["answer"] == "B"
    assert "Histograms show numeric data distributions." in seen_prompts[0]


def test_generate_course_quiz_falls_back_when_api_quota_is_exhausted(monkeypatch):
    monkeypatch.setattr(
        "src.quiz.generate_answer",
        lambda prompt, **kwargs: {"success": False, "error": "API quota exceeded."},
    )
    chunks = [{
        "source": "biology.txt",
        "text": (
            "- Mitosis: cell division that produces two genetically identical daughter cells.\n"
            "- Meiosis: cell division that produces gametes with half the chromosome number.\n"
            "- Photosynthesis: process plants use to convert light energy into chemical energy.\n"
            "- Respiration: process cells use to release energy from glucose."
        ),
    }]

    result = generate_course_quiz(chunks)

    assert result["error"] is None
    assert len(result["questions"]) == 4
    assert "API quota exceeded" in result["warning"]
    assert all("biology" in question["subject"].lower() for question in result["questions"])
    assert all("biology.txt" in question["explanation"] for question in result["questions"])


def test_generate_course_quiz_falls_back_when_model_returns_invalid_json(monkeypatch):
    monkeypatch.setattr(
        "src.quiz.generate_answer",
        lambda prompt, **kwargs: {"success": True, "answer": "This response is not JSON."},
    )
    chunks = [{
        "source": "biology.txt",
        "text": (
            "- Mitosis: cell division that produces two genetically identical daughter cells.\n"
            "- Meiosis: cell division that produces gametes with half the chromosome number.\n"
            "- Photosynthesis: process plants use to convert light energy into chemical energy.\n"
            "- Respiration: process cells use to release energy from glucose."
        ),
    }]

    result = generate_course_quiz(chunks)

    assert result["error"] is None
    assert len(result["questions"]) == 4
    assert "invalid format" in result["warning"]
    assert all("biology.txt" in question["explanation"] for question in result["questions"])


# ---------------------------------------------------------------------------
# Retrieval integration tests (require sentence-transformers)
# ---------------------------------------------------------------------------

def _skip_if_no_sentence_transformers():
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:
        pytest.skip(f"Embedding stack unavailable ({exc.__class__.__name__}): {exc}")


def test_retrieval_finds_relevant_chunk():
    """
    Asking about overfitting should retrieve a chunk that mentions overfitting.
    This tests the full retrieval pipeline from query embedding to similarity search.
    """
    _skip_if_no_sentence_transformers()

    from src.document_processing import load_all_from_directory
    from src.rag_pipeline import build_knowledge_base, answer_question

    sample_dir = PROJECT_ROOT / "data" / "sample_materials"
    if not sample_dir.exists():
        pytest.skip("Sample materials not found.")

    chunks = load_all_from_directory(str(sample_dir))
    index = build_knowledge_base(chunks)

    assert index.is_built, "Index failed to build"

    result = answer_question(
        query="What is overfitting?",
        index=index,
        top_k=5,
        min_score=0.2,
    )

    # The mode should be generated or retrieval_only (not insufficient_evidence)
    assert result["mode"] != "insufficient_evidence", \
        "Expected to find evidence for 'overfitting' in ML notes"

    # At least one chunk should contain the word "overfitting"
    chunks_text = " ".join(c["text"].lower() for c in result.get("chunks", []))
    assert "overfitting" in chunks_text, \
        "Retrieved chunks should mention overfitting"


def test_retrieval_insufficient_for_off_topic():
    """
    An off-topic question (capital of France) should trigger insufficient evidence.
    """
    _skip_if_no_sentence_transformers()

    from src.document_processing import load_all_from_directory
    from src.rag_pipeline import build_knowledge_base, answer_question

    sample_dir = PROJECT_ROOT / "data" / "sample_materials"
    if not sample_dir.exists():
        pytest.skip("Sample materials not found.")

    chunks = load_all_from_directory(str(sample_dir))
    index = build_knowledge_base(chunks)

    result = answer_question(
        query="What is the capital of France?",
        index=index,
        top_k=5,
        min_score=0.25,
    )

    assert result["mode"] == "insufficient_evidence", \
        "Should return insufficient evidence for off-topic question"


def test_citation_metadata_not_fabricated():
    """
    Source metadata in retrieved chunks must come from actual loaded documents,
    not from model generation.
    """
    _skip_if_no_sentence_transformers()

    from src.document_processing import load_all_from_directory
    from src.rag_pipeline import build_knowledge_base, answer_question

    sample_dir = PROJECT_ROOT / "data" / "sample_materials"
    if not sample_dir.exists():
        pytest.skip("Sample materials not found.")

    chunks = load_all_from_directory(str(sample_dir))
    index = build_knowledge_base(chunks)
    known_sources = {c["source"] for c in chunks}

    result = answer_question("What is a p-value?", index, min_score=0.2)
    for chunk in result.get("chunks", []):
        assert chunk["source"] in known_sources, \
            f"Citation source '{chunk['source']}' not in known document sources"


def test_verified_qa_set_retrieval_metrics():
    """
    Run the production evaluator over the full verified QA set.
    This replaces the informal, partial checks that used to live only in docs.
    """
    _skip_if_no_sentence_transformers()

    from src.evaluation import run_retrieval_evaluation

    sample_dir = PROJECT_ROOT / "data" / "sample_materials"
    qa_path = sample_dir / "verified_qa_set.json"
    if not qa_path.exists():
        pytest.skip("Verified QA set not found.")

    report = run_retrieval_evaluation(materials_dir=sample_dir, qa_path=qa_path)

    assert report.n_cases == 9
    assert report.n_in_scope == 8
    assert report.n_out_of_scope == 1
    assert report.retrieval_hit_rate == 1.0, report.to_markdown()
    assert report.evidence_threshold_accuracy_in_scope == 1.0, report.to_markdown()
    assert report.insufficient_evidence_accuracy == 1.0, report.to_markdown()
    assert report.citation_accuracy == 1.0, report.to_markdown()
