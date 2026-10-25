"""
embeddings.py
=============
STEP 4 and STEP 5 of the RAG pipeline: Embedding Generation and Vector Indexing.

WHAT IS AN EMBEDDING?
----------------------
An embedding is a list of numbers (a vector) that represents the meaning of a
piece of text. The key property is: two texts with similar meaning will have
vectors that are close together in the vector space, even if the exact words differ.

For example:
  "How does a decision tree work?" 
  "Explain the mechanism of a decision tree."
Both map to nearly the same vector, so a search for the first question will
retrieve text relevant to the second question too.

This is fundamentally different from keyword search, which would only find
documents containing the exact words used in the query.

WHICH MODEL DO WE USE?
-----------------------
We use sentence-transformers/all-MiniLM-L6-v2 — a small, fast, and high-quality
embedding model. It:
  - Runs completely locally on your CPU (no internet required after downloading).
  - Produces 384-dimensional vectors.
  - Was trained specifically to create sentence-level embeddings that capture meaning.
  - Is one of the most widely recommended models for RAG applications.

HOW DOES INDEXING WORK?
-----------------------
We store the embeddings in a plain NumPy array (a matrix).
  - Each row = the embedding of one chunk.
  - The row index corresponds to the chunk's position in our chunk list.

When a user asks a question:
  1. We embed the question into a 384-dimensional vector.
  2. We compute the cosine similarity between the question vector and every row.
  3. We return the chunks corresponding to the highest similarity scores.

WHY COSINE SIMILARITY?
-----------------------
Cosine similarity measures the angle between two vectors, not their length.
  - Two vectors pointing in the same direction → cosine = 1.0 (identical meaning)
  - Two perpendicular vectors → cosine = 0.0 (unrelated meaning)
  - Two opposite vectors → cosine = -1.0 (opposite meaning)

For sentence embeddings, cosine similarity is preferred over Euclidean distance
because it is scale-invariant — it does not depend on how long the text is.
"""

import re
from typing import Optional

import numpy as np

from src.config import DEFAULT_CONFIG

# Lazy import: we only import sentence_transformers when this module is actually used.
# This avoids a slow startup if the user just runs tests that don't need embeddings.
_model = None  # Module-level cache for the loaded model
_last_embedding_error: Optional[str] = None


def get_embedding_model(model_name: str = DEFAULT_CONFIG.embedding_model):
    """
    Load and cache the sentence-transformer embedding model.

    WHY CACHE IT?
    -------------
    Loading a model from disk takes a few seconds. If we loaded it fresh on
    every embedding call, the application would be very slow. We keep a single
    loaded model instance in the module-level `_model` variable and reuse it.

    Parameters
    ----------
    model_name : HuggingFace model identifier. Defaults to all-MiniLM-L6-v2.

    Returns
    -------
    A SentenceTransformer model object, or None if the library is not installed.
    """
    global _last_embedding_error, _model
    _last_embedding_error = None
    if _model is not None:
        return _model  # Return the cached model

    try:
        from sentence_transformers import SentenceTransformer
        print(f"[INFO] Loading embedding model: {model_name}")
        print("       (First run downloads the model — takes ~1 minute. Subsequent runs are instant.)")
        _model = SentenceTransformer(model_name)
        print(f"[INFO] Embedding model loaded. Vector dimensions: {_model.get_sentence_embedding_dimension()}")
        return _model
    except ImportError as e:
        _last_embedding_error = f"sentence-transformers is not installed: {e}"
        print("[ERROR] sentence-transformers is not installed.")
        print("        Run:  python -m pip install -r requirements.txt")
        return None
    except Exception as e:
        _last_embedding_error = f"{type(e).__name__}: {e}"
        print(f"[ERROR] Failed to load embedding model: {e}")
        return None


def get_embedding_error() -> Optional[str]:
    """Return the most recent embedding-model or encoding error, if any."""
    return _last_embedding_error


def embed_texts(texts: list[str], model_name: str = DEFAULT_CONFIG.embedding_model) -> Optional[np.ndarray]:
    """
    Convert a list of text strings into a 2D NumPy array of embeddings.

    Each row in the output array is the embedding of the corresponding input text.
    If we embed 50 chunks, the output shape is (50, 384).

    Parameters
    ----------
    texts      : A list of text strings to embed.
    model_name : The embedding model to use.

    Returns
    -------
    A NumPy array of shape (len(texts), embedding_dimension), or None on failure.

    WHAT SHAPE MEANS
    ----------------
    Shape (50, 384) means:
      - 50 rows → one per text input.
      - 384 columns → each text is represented by 384 numbers.
    The 384 numbers together encode the "meaning" of the text in a high-dimensional
    geometric space where similar meanings cluster together.
    """
    if not texts:
        return None

    model = get_embedding_model(model_name)
    if model is None:
        return None

    try:
        # encode() does the actual computation.
        # show_progress_bar=True prints a progress bar for large batches.
        # convert_to_numpy=True returns a NumPy array (the default is a PyTorch tensor).
        embeddings = model.encode(
            texts,
            show_progress_bar=len(texts) > 20,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize so cosine similarity = dot product
        )
        return embeddings  # shape: (len(texts), 384)
    except Exception as e:
        global _last_embedding_error
        _last_embedding_error = f"{type(e).__name__}: {e}"
        print(f"[ERROR] Embedding failed: {e}")
        return None


def embed_query(query: str, model_name: str = DEFAULT_CONFIG.embedding_model) -> Optional[np.ndarray]:
    """
    Embed a single query string and return a 1D vector.

    This is identical to embed_texts but returns a 1D array (shape: (384,))
    instead of a 2D array, for convenience when computing similarity against
    a pre-built index.

    Parameters
    ----------
    query : The user's question as a plain string.

    Returns
    -------
    A 1D NumPy array of shape (384,), or None on failure.
    """
    result = embed_texts([query], model_name)
    if result is not None:
        return result[0]  # Unwrap the single row from the (1, 384) array
    return None


def _keyword_overlap_score(query: str, text: str) -> float:
    """Lightweight lexical boost for short, semantically relevant queries.

    Embedding similarity is the primary signal, but short questions such as
    "What is RAG?" can still match a relevant document weakly because the
    abbreviation is uncommon or the wording differs. Adding a small lexical
    overlap term helps retrieve the right chunk while keeping off-topic queries
    below the evidence threshold.
    """
    if not query or not text:
        return 0.0

    query_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if len(token) > 2 and token not in {"what", "when", "where", "why", "which", "with", "from", "into"}
    }
    if not query_terms:
        return 0.0

    doc_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2
    }
    if not doc_terms:
        return 0.0

    overlap = len(query_terms & doc_terms)
    return (overlap / max(len(query_terms), 1)) * 0.2


# ---------------------------------------------------------------------------
# In-memory vector index
# ---------------------------------------------------------------------------

class VectorIndex:
    """
    A simple in-memory vector index backed by NumPy and scikit-learn cosine similarity.

    WHY NOT FAISS OR CHROMADB?
    --------------------------
    FAISS and ChromaDB are excellent for production systems with millions of vectors.
    For our use case (hundreds to a few thousand chunks), plain NumPy is:
      - Simpler to understand and debug.
      - No additional installation required beyond NumPy.
      - Fast enough — computing cosine similarity over 1000 vectors takes <1ms.
      - No external database process to manage.

    We can always upgrade to FAISS later if performance becomes a concern.

    ATTRIBUTES
    ----------
    chunks         : The list of chunk dicts whose text was embedded.
    embeddings     : A (num_chunks, 384) NumPy array. Row i = embedding of chunk i.
    is_built       : Whether the index has been populated.
    """

    def __init__(self):
        self.chunks: list[dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.is_built: bool = False

    def build(self, chunks: list[dict], model_name: str = DEFAULT_CONFIG.embedding_model) -> bool:
        """
        Embed all chunks and build the searchable index.

        Parameters
        ----------
        chunks     : List of chunk dicts from document_processing.py.
        model_name : Embedding model to use.

        Returns
        -------
        True if the index was built successfully, False otherwise.
        """
        if not chunks:
            print("[WARNING] No chunks provided to build index.")
            return False

        print(f"[INFO] Building vector index for {len(chunks)} chunks...")

        # Extract just the text from each chunk for embedding
        texts = [chunk["text"] for chunk in chunks]

        embeddings = embed_texts(texts, model_name)
        if embeddings is None:
            print("[ERROR] Could not generate embeddings. Index not built.")
            return False

        self.chunks = chunks
        self.embeddings = embeddings
        self.is_built = True

        print(f"[INFO] Vector index built. Shape: {embeddings.shape}")
        print(f"       Each of {embeddings.shape[0]} chunks is represented as a {embeddings.shape[1]}-dimensional vector.")
        return True

    def search(self, query: str, top_k: int = DEFAULT_CONFIG.top_k, model_name: str = DEFAULT_CONFIG.embedding_model) -> list[dict]:
        """
        Find the top_k most semantically similar chunks to the query.

        HOW IT WORKS
        ------------
        1. Embed the query into a vector q of shape (384,).
        2. Compute the cosine similarity between q and every row in self.embeddings.
           Because we normalized the embeddings at creation time, cosine similarity
           is simply the dot product: q · chunk_embedding.
        3. Sort by similarity score (highest first).
        4. Return the top_k chunks with their similarity scores attached.

        Parameters
        ----------
        query    : The user's question string.
        top_k    : How many chunks to return (default 5).
                   More chunks = more context for the LLM but also more noise.
                   5 is a good default for our domain.

        Returns
        -------
        A list of at most top_k chunk dicts, each with an added "score" key.
        The list is sorted by score, highest first.
        Returns an empty list if the index has not been built.
        """
        if not self.is_built:
            print("[ERROR] Index has not been built. Call build() first.")
            return []

        # Step 1: Embed the query
        query_vector = embed_query(query, model_name)
        if query_vector is None:
            return []

        # Step 2: Compute cosine similarity
        # self.embeddings has shape (N, 384), query_vector has shape (384,).
        # Matrix-vector dot product gives us shape (N,) — one score per chunk.
        # This works because we normalized the vectors at embedding time.
        scores = np.dot(self.embeddings, query_vector)  # shape: (num_chunks,)

        # Step 3: Sort by score, descending
        # np.argsort returns indices that would sort the array ascending.
        # [::-1] reverses to descending order.
        ranked_indices = np.argsort(scores)[::-1]

        # Step 4: Take the top_k results and attach their scores.
        # We apply a small lexical-overlap boost to short or abbreviation-heavy
        # questions (e.g. 'What is RAG?') so relevant material is not filtered
        # out just because the embedding similarity is close to the threshold.
        results = []
        for idx in ranked_indices[:top_k]:
            chunk = dict(self.chunks[idx])
            semantic_score = float(scores[idx])
            lexical_boost = _keyword_overlap_score(query, chunk.get("text", ""))
            chunk["score"] = semantic_score + lexical_boost
            results.append(chunk)

        return results

    def get_stats(self) -> dict:
        """
        Return basic statistics about the current index state.
        Used for display in the Streamlit sidebar.
        """
        if not self.is_built:
            return {"built": False, "num_chunks": 0, "vector_dimensions": 0}
        return {
            "built": True,
            "num_chunks": len(self.chunks),
            "vector_dimensions": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "sources": list(set(c["source"] for c in self.chunks)),
        }
