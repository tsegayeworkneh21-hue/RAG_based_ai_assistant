# Engineering Decision Log

This document records the important technical decisions made during the development of ExamPilot AI, along with the alternatives considered and the reasons for the choices made.

---

## Decision 1: Embedding Model — `all-MiniLM-L6-v2`

**Decision:** Use `sentence-transformers/all-MiniLM-L6-v2` for generating embeddings.

**Alternatives considered:**
- `text-embedding-ada-002` (OpenAI): requires an API call per embedding, incurs cost, adds network dependency.
- `all-mpnet-base-v2`: higher quality but slower and uses more memory.
- `paraphrase-multilingual-MiniLM-L12-v2`: needed only for multilingual support.
- Custom fine-tuned embedding: out of scope for the timeline.

**Reason for choice:** MiniLM-L6-v2 runs entirely locally (no internet call for inference), produces strong 384-dimensional embeddings suitable for this domain, and is fast enough to embed all sample materials in under 60 seconds on a CPU. Its quality on semantic textual similarity benchmarks is excellent for its size.

---

## Decision 2: Vector Store — NumPy Array Instead of a Dedicated Vector Database

**Decision:** Store embeddings in a plain NumPy matrix and use dot-product similarity (after L2 normalization).

**Alternatives considered:**
- FAISS: optimised for millions of vectors; adds complexity for our scale.
- ChromaDB: adds a database process and persistence layer.
- Pinecone (cloud): adds external dependency, cost, and account setup.
- Weaviate: production-scale, far beyond project needs.

**Reason for choice:** Our corpus is ~100 chunks from sample materials, and potentially a few hundred if the student uploads additional files. NumPy dot-product search over 500 vectors completes in under 1 millisecond. Introducing a vector database would add complexity without measurable benefit at this scale. This also makes the code easier to read and teach.

---

## Decision 3: Chunk Size 800 Characters with 150 Overlap

**Decision:** Fixed-size character chunking, chunk_size=800, overlap=150.

**Alternatives considered:**
- Sentence-level chunking: cleaner but fragile with bullet points and equations.
- Paragraph-level chunking: inconsistent sizes depending on document formatting.
- 512 characters: too small for multi-sentence concept explanations.
- 1500 characters: risks exceeding embedding model's effective window.

**Reason for choice:** 800 characters (~130 words) is large enough to capture a complete concept explanation, small enough to remain focused, and safely within the 256-token effective input size of all-MiniLM-L6-v2 (which internally truncates inputs). The 150-character overlap ensures that concept sentences spanning a boundary are captured by at least one chunk.

---

## Decision 4: Minimum Similarity Threshold — 0.25

**Decision:** Reject retrieved chunks with cosine similarity below 0.25.

**Alternatives considered:**
- No threshold (always return top-k): would produce irrelevant context for off-topic questions.
- Threshold of 0.4: too strict, rejects moderately relevant content.
- Threshold of 0.15: too permissive, passes clearly unrelated content.

**Reason for choice:** Empirically tested on the verified QA set. Questions within the covered subjects score 0.4–0.85. The geography question ("capital of France") scored below 0.20 against all chunks. 0.25 provides a clean separation without manual tuning for each subject area.

---

## Decision 5: Gemini 1.5 Flash as the LLM

**Decision:** Use `gemini-1.5-flash` as the default generation model.

**Alternatives considered:**
- `gemini-1.5-pro`: higher quality, higher cost, slower.
- GPT-4o (OpenAI): excellent quality, requires paid API.
- Local LLM (Ollama, llama.cpp): removes API dependency, but requires 8+ GB RAM and complex setup on Windows.
- Claude (Anthropic): requires separate account and API.

**Reason for choice:** Gemini Flash offers a generous free tier, low latency, and a 1M-token context window. The official Python client is well-maintained. The free tier is sufficient for demonstration and evaluation within the project timeline.

---

## Decision 6: Streamlit for the UI

**Decision:** Use Streamlit as the only UI framework.

**Alternatives considered:**
- FastAPI + React: requires JavaScript knowledge, separate frontend, deployment complexity.
- Gradio: similar to Streamlit but less feature-rich for multi-tab layouts.
- Flask + Jinja2: requires HTML/CSS knowledge.

**Reason for choice:** Streamlit allows building a professional, interactive UI entirely in Python. The student already has Python experience. Streamlit Community Cloud provides free deployment directly from GitHub. Session state management is adequate for our use case.

---

## Decision 7: Retrieval-Only Fallback Mode

**Decision:** When no API key is configured, display retrieved passages in a structured format rather than showing an error.

**Reason for choice:** The most important part of the RAG pipeline is retrieval — showing relevant evidence. The LLM layer adds explanation but the core value (finding relevant passages) still works without an API key. This makes the application demonstrable even in an environment without internet access or when the free quota is exceeded. Labelling it clearly as "Retrieval-Only Mode" maintains honesty about what is actually happening.

---

## Decision 8: Plain Python Dicts for Chunks (Not a Dataclass or ORM)

**Decision:** Represent each chunk as a plain Python dict: `{"text": ..., "source": ..., "page": ..., "chunk_index": ...}`.

**Reason for choice:** Dicts are transparent, printable, JSON-serializable, and require no class definition. The chunk structure is simple (4 fields). A dataclass would add ceremony without benefit at this scale. This also makes the code easier to explain to a student.
