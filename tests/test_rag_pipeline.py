"""Tests for user-facing generation failures in the RAG pipeline."""

from src import rag_pipeline


def test_raw_gemini_504_is_normalized_in_retrieval_fallback(monkeypatch):
    raw_error = "504 Deadline expired before operation could complete."
    index = object()

    def fake_retrieve(query, actual_index, top_k, min_score):
        assert query == "Question?"
        assert actual_index is index
        assert top_k == rag_pipeline.DEFAULT_CONFIG.top_k
        assert min_score == rag_pipeline.DEFAULT_CONFIG.min_score
        return {
            "chunks": [{"text": "Relevant course passage."}],
            "has_evidence": True,
            "all_results": [],
        }

    def fake_build_full_prompt(query, result):
        assert query == "Question?"
        assert result["has_evidence"]
        return "prompt"

    monkeypatch.setattr(rag_pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag_pipeline, "build_full_prompt", fake_build_full_prompt)
    monkeypatch.setattr(rag_pipeline, "is_api_available", lambda: True)
    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        lambda prompt: {
            "success": False,
            "error": raw_error if prompt == "prompt" else "unexpected prompt",
        },
    )
    monkeypatch.setattr(
        rag_pipeline,
        "format_retrieval_only_response",
        lambda result, generation_error=None: (
            f"Retrieved passage fallback: {generation_error}"
            if result["has_evidence"] and generation_error
            else "No evidence"
        ),
    )

    result = rag_pipeline.answer_question("Question?", index=index)

    assert result["mode"] == "retrieval_only"
    assert result["answer"].startswith("Retrieved passage fallback:")
    assert "Gemini returned a timeout/deadline error (504)" in result["error"]
    assert raw_error not in result["error"]
