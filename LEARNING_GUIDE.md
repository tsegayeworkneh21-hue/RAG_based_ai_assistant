# Learning Guide — ExamPilot AI
## What I Built, How It Works, and How to Defend It

---

## Day 1 — What I Learned

### Concept 1: What RAG actually means

RAG stands for **Retrieval-Augmented Generation**. The word "augmented" is the key.

A plain LLM (like Gemini or ChatGPT) generates answers from its training data. If you ask it about your specific university notes, it might answer from general internet knowledge instead — or worse, confidently make something up. This is called **hallucination**.

RAG fixes this by adding a retrieval step:

```
Student question
      ↓
[RETRIEVE] Search actual documents → get relevant passages
      ↓
[AUGMENT]  Add those passages to the LLM prompt as context
      ↓
[GENERATE] LLM generates explanation from those passages only
```

The LLM is not answering from its imagination — it is explaining what it found in your documents.

---

### Concept 2: What an embedding is

An embedding is a list of numbers that represents the meaning of a piece of text.

For example:
- `"overfitting"` → `[0.12, -0.45, 0.78, ..., 0.03]` (384 numbers)
- `"model memorises training data"` → `[0.13, -0.44, 0.79, ..., 0.04]`

These two phrases mean almost the same thing, so their vectors are almost identical.

Meanwhile:
- `"capital of France"` → `[0.91, 0.02, -0.33, ..., -0.88]` — very different from overfitting vectors.

This property (similar meaning → similar vectors) is what makes semantic search possible. We can find relevant text even when the student's question uses different words than the document.

**What an embedding is NOT:**
- It is not the words stored as ASCII codes.
- It is not a keyword frequency count (like TF-IDF).
- It cannot be read back as text — it is a compressed representation of meaning.

---

### Concept 3: Cosine similarity

Once we have vectors, how do we measure "how similar"?

We use **cosine similarity**, which measures the angle between two vectors.

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

- `1.0` = perfect match (same direction)
- `0.0` = unrelated (perpendicular)
- `-1.0` = opposite meaning

We normalize our vectors (make them length 1) at embedding time, so the formula simplifies to just the dot product: `A · B`.

In NumPy:
```python
scores = np.dot(embeddings, query_vector)
# embeddings shape: (N, 384)
# query_vector shape: (384,)
# result shape: (N,)  — one score per chunk
```

This single line computes the similarity between the query and all N chunks at once.

---

### Concept 4: Chunking

Why do we split documents into chunks?

1. **Embedding models have a maximum input size.** `all-MiniLM-L6-v2` processes up to 256 tokens (~200 words) at a time. A 50-page textbook cannot fit.

2. **Smaller chunks are more precise.** If we embed an entire document, the vector represents the average of all topics. A chunk about overfitting will have a much stronger overfitting signal.

3. **Retrieval returns specific passages, not whole documents.** We want to show the student the exact relevant paragraph, not the entire file.

Our settings:
- `chunk_size = 800` characters (~130 words) — captures a complete concept explanation.
- `overlap = 150` characters — prevents concepts from being split at chunk boundaries.

---

### Concept 5: The pipeline stages (what each file does)

| File | Stage | Input | Output |
|------|-------|-------|--------|
| `document_processing.py` | Load + Extract + Chunk | `.txt` or `.pdf` file path | List of chunk dicts |
| `embeddings.py` | Embed + Index | List of chunk dicts | VectorIndex (NumPy matrix) |
| `retrieval.py` | Retrieve + Prompt | Query string + VectorIndex | Retrieved chunks + Prompt |
| `llm_provider.py` | Generate | Full prompt string | Answer text |
| `rag_pipeline.py` | Orchestrate | Query + Index | Final response dict |

---

## Important Functions — Code Walkthrough

### `chunk_pages()` in document_processing.py

```python
def chunk_pages(pages, chunk_size=800, overlap=150):
    chunks = []
    chunk_index = 0

    for page_dict in pages:             # For each page in the document
        text = page_dict["text"]        # Get the full text of this page
        source = page_dict["source"]    # "machine_learning_notes.txt"
        page = page_dict["page"]        # Page number (or None for .txt files)

        start = 0                       # Start of the current chunk window

        while start < len(text):        # Keep chunking until we reach the end
            end = start + chunk_size    # End of this chunk window
            chunk_text = text[start:end]  # Extract the text

            # Try to find a sentence boundary near the end
            # (so we don't cut mid-sentence)
            ...

            chunks.append(make_chunk(
                text=chunk_text,
                source=source,
                page=page,
                chunk_index=chunk_index
            ))
            chunk_index += 1

            # Move forward by (chunk_size - overlap)
            # This creates the overlap: the next chunk starts 150 chars before
            # where this one ended
            start += (end - start - overlap)

    return chunks
```

**What would happen if we removed the overlap?**  
A concept like "The bias-variance trade-off..." that spans the end of one 800-char window and the start of the next would be split. A question asking about bias-variance might only retrieve a fragment.

---

### `VectorIndex.search()` in embeddings.py

```python
def search(self, query, top_k=5):
    # Step 1: Convert the query string to a 384-dim vector
    query_vector = embed_query(query)       # shape: (384,)

    # Step 2: Compute cosine similarity with every stored chunk
    scores = np.dot(self.embeddings, query_vector)  # shape: (N,)
    # self.embeddings is our (N, 384) matrix
    # The dot product is cosine similarity because we normalized at build time

    # Step 3: Sort indices from highest score to lowest
    ranked_indices = np.argsort(scores)[::-1]
    # argsort gives ASCENDING order, [::-1] reverses to DESCENDING

    # Step 4: Return the top_k chunks with their scores
    results = []
    for idx in ranked_indices[:top_k]:
        chunk = dict(self.chunks[idx])       # Copy the chunk
        chunk["score"] = float(scores[idx])  # Attach the score
        results.append(chunk)

    return results
```

**Key insight:** The entire "search" is three NumPy operations — embed, dot product, argsort. This is why it's fast enough to run in real-time.

---

### `retrieve()` in retrieval.py — the honesty function

```python
def retrieve(query, index, top_k=5, min_score=0.25):
    all_results = index.search(query, top_k=top_k)

    # Filter: only keep chunks above the minimum relevance score
    relevant_chunks = [
        chunk for chunk in all_results
        if chunk.get("score", 0) >= min_score
    ]

    return {
        "chunks": relevant_chunks,
        "has_evidence": len(relevant_chunks) > 0,  # True only if something passed
        ...
    }
```

**Why 0.25?**  
Empirically tested. Questions about the covered subjects score 0.4–0.85. The geography question scored below 0.20. The 0.25 threshold cleanly separates relevant from irrelevant retrieval.

**What if we removed this threshold?**  
The system would always retrieve chunks (even if totally unrelated) and pass them to the LLM. The LLM would generate a confident-sounding but unsupported answer. The system would hallucinate.

---

## Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|---------|
| `ModuleNotFoundError: sentence_transformers` | Not installed | `pip install sentence-transformers` |
| `ModuleNotFoundError: fitz` | PyMuPDF not installed | `pip install pymupdf` |
| `ModuleNotFoundError: dotenv` | python-dotenv not installed | `pip install python-dotenv` |
| Empty response from Gemini | API key missing | Add `GEMINI_API_KEY=...` to `.env` |
| "Index has not been built" | Skipped the Build step | Click "Build Knowledge Base" in the sidebar |
| Slow on first run | Model download (one time only) | Wait ~60 seconds; subsequent runs are instant |
| PDF shows no text | Scanned image PDF | OCR (Tesseract) required — not in scope |

---

## Instructor Questions and Answers

**Q: What is the difference between RAG and fine-tuning?**  
A: Fine-tuning modifies the model's weights by training on new data. The model permanently incorporates the new knowledge, but you need GPU resources and training time, and the knowledge is "baked in" (hard to update). RAG adds a retrieval layer in front of an unmodified model. You update the knowledge by changing the documents in the index — no retraining required. RAG is also more transparent: you can inspect exactly which passages were retrieved.

**Q: How do you handle the case where the model makes something up?**  
A: Three safeguards: (1) the system prompt explicitly instructs the LLM to answer only from retrieved evidence, (2) the similarity threshold prevents irrelevant passages from becoming context, (3) the citation panel shows the actual retrieved text so the student can check whether the answer matches the source. The system cannot guarantee zero hallucination, but it makes hallucination detectable.

**Q: Why do you use cosine similarity instead of just keyword matching?**  
A: Keyword matching (like searching for the exact word "overfitting") would miss queries like "when a model memorises its training data" even though they mean the same thing. Cosine similarity in embedding space captures semantic meaning, not exact word overlap.

**Q: What is a vector dimension, and why 384?**  
A: A dimension is one number in the vector. `all-MiniLM-L6-v2` was trained to produce 384-dimensional vectors — this is an architecture decision by the model creators. More dimensions = more expressive but slower. 384 is a balance point: fast to compute, strong enough to capture sentence-level meaning.

**Q: What would you do to improve this system?**  
A: (1) Add OCR for scanned PDFs using Tesseract. (2) Add index persistence so the index doesn't rebuild on every restart. (3) Evaluate on a larger, independently-created question set. (4) Add re-ranking of retrieved results using a cross-encoder model. (5) Support multilingual materials with a multilingual embedding model.

---

## Example: What the Prompt Looks Like

When you ask "What is overfitting?", the system constructs and sends this to Gemini:

```
You are an AI study assistant helping Data Science students prepare for their exit examination.
[... system instructions ...]

=== RETRIEVED EVIDENCE FROM COURSE MATERIALS ===

[SOURCE 1] machine_learning_notes.txt (Page None) | Relevance: 0.821
------------------------------------------------------------
Overfitting occurs when a model learns the training data too well, including its noise
and random fluctuations. An overfit model achieves high accuracy on training data but
performs poorly on new, unseen data. It has memorised rather than generalised.

[SOURCE 2] machine_learning_notes.txt (Page None) | Relevance: 0.763
------------------------------------------------------------
The bias-variance trade-off:
  - High bias → underfitting: the model makes strong assumptions and misses patterns.
  - High variance → overfitting: the model is too sensitive to training data noise.
...
==================================================

[STUDENT QUESTION]
What is overfitting?

[INSTRUCTION]
Answer the student's question using ONLY the retrieved evidence above.
Cite sources using [SOURCE 1], [SOURCE 2], etc. for each point you make.
```

The LLM generates its answer from this prompt — not from general internet knowledge.
