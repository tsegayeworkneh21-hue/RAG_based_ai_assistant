"""
rag_pipeline.py
===============
The orchestrator: ties together all pipeline stages into a single callable interface.

WHAT THIS MODULE DOES
---------------------
This module does NOT implement any logic itself. Its job is to connect:
  document_processing.py → embeddings.py → retrieval.py → llm_provider.py

and expose a clean, simple interface that app.py can call without needing
to know the internals of each stage.

Think of this as the "conductor" of an orchestra: each instrument (module)
knows how to play its part; the conductor decides when and in what order.

THE COMPLETE RAG FLOW
---------------------
1. build_knowledge_base(chunks)
   → Embed chunks and build the vector index.

2. answer_question(query, index)
   → Retrieve relevant chunks.
   → Build the full prompt.
   → Call the LLM (or fall back to retrieval-only mode).
   → Return a structured response with the answer and citations.

RESPONSE STRUCTURE
------------------
Every call to answer_question() returns a dict:
{
    "answer":         str — the generated answer text (or retrieval-only text),
    "mode":           str — "generated", "retrieval_only", "insufficient_evidence", or "error",
    "has_evidence":   bool — whether relevant chunks were found,
    "chunks":         list — the retrieved chunk dicts (for citation display),
    "query":          str — the original question,
    "error":          str|None — error message if something went wrong,
    "model":          str|None — the model used, if generation succeeded,
}

Having a consistent response structure means app.py can always check
result["mode"] and result["answer"] without handling different cases.
"""

from src.config import DEFAULT_CONFIG
from src.embeddings import VectorIndex
from src.retrieval import retrieve, build_full_prompt
from src.llm_provider import (
    generate_answer,
    format_retrieval_only_response,
    format_generation_error,
    is_api_available,
)


def build_knowledge_base(
    chunks: list[dict],
    model_name: str = DEFAULT_CONFIG.embedding_model,
) -> VectorIndex:
    """
    Embed the provided chunks and return a built VectorIndex.

    Parameters
    ----------
    chunks     : List of chunk dicts from document_processing.py.
    model_name : Embedding model identifier (PipelineConfig.embedding_model).

    Returns
    -------
    A VectorIndex. Check index.is_built to verify success.
    """
    index = VectorIndex()
    index.build(chunks, model_name=model_name)
    return index


def answer_question(
    query: str,
    index: VectorIndex,
    top_k: int = DEFAULT_CONFIG.top_k,
    min_score: float = DEFAULT_CONFIG.min_score,
) -> dict:
    """
    Run the complete RAG pipeline for a single question.

    STEP-BY-STEP WHAT HAPPENS
    --------------------------
    1. Call retrieve() to find semantically similar chunks.
    2. If no relevant chunks found → return "insufficient_evidence" response.
    3. If API key is available → build the full prompt and call generate_answer().
    4. If API key is NOT available → call format_retrieval_only_response().
    5. Package everything (answer, chunks, mode, citations) into a response dict.

    Parameters
    ----------
    query     : The student's question.
    index     : A built VectorIndex from build_knowledge_base().
    top_k     : Number of chunks to retrieve.
    min_score : Minimum similarity threshold for evidence.

    Returns
    -------
    A response dict (see module docstring for structure).
    """
    # ---- STEP 6: Retrieval ----
    retrieval_result = retrieve(query, index, top_k=top_k, min_score=min_score)
    retrieval_result["query"] = query  # Attach the query for downstream use

    # ---- Handle: Index not built ----
    if "error" in retrieval_result and not retrieval_result.get("has_evidence"):
        return {
            "answer": f"System error: {retrieval_result['error']}",
            "mode": "error",
            "has_evidence": False,
            "chunks": [],
            "query": query,
            "error": retrieval_result.get("error"),
            "model": None,
        }

    # ---- Handle: No evidence found ----
    if not retrieval_result["has_evidence"]:
        # Check if there were ANY results (even below threshold) to give a better message
        all_results = retrieval_result.get("all_results", [])
        best_score = all_results[0].get("score", 0) if all_results else 0

        insufficient_msg = (
            "The available course materials do not contain sufficient information "
            "to answer this question.\n\n"
            "The materials currently cover: Machine Learning, Introduction to "
            "Artificial Intelligence, and Inferential Statistics.\n\n"
            f"*(Best match found had relevance score: {best_score:.3f} — "
            f"threshold is {min_score})*"
        )

        return {
            "answer": insufficient_msg,
            "mode": "insufficient_evidence",
            "has_evidence": False,
            "chunks": [],
            "all_results": all_results,
            "query": query,
            "error": None,
            "model": None,
        }

    # ---- STEP 7: Context Construction ----
    full_prompt = build_full_prompt(query, retrieval_result)

    # ---- STEP 8: Generation or Retrieval-Only ----
    if is_api_available():
        # We have an API key — call the LLM
        llm_result = generate_answer(full_prompt)

        if llm_result["success"]:
            return {
                "answer": llm_result["answer"],
                "mode": "generated",
                "has_evidence": True,
                "chunks": retrieval_result["chunks"],
                "query": query,
                "error": None,
                "model": llm_result["model"],
                "prompt_used": full_prompt,  # Stored for transparency/debugging
            }
        else:
            # LLM call failed (e.g., bad API key, quota exceeded)
            # Fall back to retrieval-only and include the error
            generation_error = format_generation_error(str(llm_result["error"]))
            retrieval_only_answer = format_retrieval_only_response(
                retrieval_result,
                generation_error=generation_error,
            )
            return {
                "answer": retrieval_only_answer,
                "mode": "retrieval_only",
                "has_evidence": True,
                "chunks": retrieval_result["chunks"],
                "query": query,
                "error": f"LLM generation failed: {generation_error}",
                "model": None,
            }
    else:
        # No API key — retrieval-only mode
        retrieval_only_answer = format_retrieval_only_response(retrieval_result)
        return {
            "answer": retrieval_only_answer,
            "mode": "retrieval_only",
            "has_evidence": True,
            "chunks": retrieval_result["chunks"],
            "query": query,
            "error": "No GEMINI_API_KEY configured. Showing retrieved passages only.",
            "model": None,
        }


def get_pipeline_status(index: VectorIndex) -> dict:
    """
    Return a summary of the current pipeline state for display in the UI.

    Parameters
    ----------
    index : The VectorIndex to inspect.

    Returns
    -------
    A dict with human-readable status information.
    """
    index_stats = index.get_stats()
    api_available = is_api_available()

    return {
        "index_built": index_stats["built"],
        "num_chunks": index_stats.get("num_chunks", 0),
        "vector_dimensions": index_stats.get("vector_dimensions", 0),
        "sources_loaded": index_stats.get("sources", []),
        "llm_available": api_available,
        "mode": "Full RAG (Retrieval + Generation)" if api_available else "Retrieval-Only Mode",
    }
