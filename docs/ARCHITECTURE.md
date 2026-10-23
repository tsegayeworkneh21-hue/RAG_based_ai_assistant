# Architecture Overview

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STUDENT BROWSER / VS CODE                       │
│                              (Streamlit UI)                              │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ question text
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          app.py  (UI layer)                              │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │  Home    │  │  Q&A Panel   │  │  Quiz Panel  │  │ Pipeline Debug  │ │
│  └──────────┘  └──────┬───────┘  └──────────────┘  └─────────────────┘ │
└──────────────────────┬┘────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    rag_pipeline.py  (Orchestrator)                       │
│                                                                          │
│  build_knowledge_base(chunks)  ──────────────────────────┐              │
│  answer_question(query, index)  ─────────────────────┐   │              │
└──────────────────────────────────────────────────────┼───┼──────────────┘
                                                       │   │
                    ┌──────────────────────────────────┘   │
                    │                                       │
                    ▼                                       ▼
┌───────────────────────────┐          ┌────────────────────────────────┐
│     retrieval.py          │          │     embeddings.py              │
│  retrieve(query, index)   │          │  VectorIndex.build(chunks)     │
│  build_context_string()   │          │  VectorIndex.search(query)     │
│  build_full_prompt()      │          │  embed_texts(texts)            │
│  build_system_prompt()    │          │  embed_query(query)            │
└──────────┬────────────────┘          └────────────────────────────────┘
           │                                        ▲
           │                            ┌───────────┘
           ▼                            │
┌─────────────────────────┐   ┌─────────────────────────────────────────┐
│    llm_provider.py      │   │    document_processing.py               │
│  generate_answer(prompt)│   │  load_txt_file(path)                    │
│  is_api_available()     │   │  load_pdf_file(path)                    │
│  format_retrieval_only()│   │  load_uploaded_file(bytes, name)        │
└──────────┬──────────────┘   │  chunk_pages(pages)                     │
           │                   │  load_all_from_directory(dir)          │
           ▼                   └──────────────────────────────────────┘
┌──────────────────────┐
│  Google Gemini API   │  ←── GEMINI_API_KEY from .env
│  (external service)  │
└──────────────────────┘
```

## Data Flow

### Indexing (one-time, triggered by "Build Knowledge Base" button)

```
PDF/TXT files
    │
    ▼
load_txt_file() / load_pdf_file()
    │
    │  Output: List[{text, page, source}]  — one entry per page
    ▼
chunk_pages(pages, chunk_size=800, overlap=150)
    │
    │  Output: List[{text, source, page, chunk_index}]
    ▼
embed_texts(texts)  — calls sentence-transformers model
    │
    │  Output: NumPy array shape (N, 384)
    ▼
VectorIndex.build()
    │
    │  Stores: self.chunks (list), self.embeddings (N×384 matrix)
    ▼
[Index stored in st.session_state.index]
```

### Querying (triggered by each student question)

```
Student question (string)
    │
    ▼
embed_query(query)  — 384-dimensional vector
    │
    ▼
np.dot(embeddings, query_vector)  — cosine similarity for all N chunks
    │
    ▼
Sort by score → filter by min_score threshold → top_k chunks
    │
    ▼
build_context_string()  — format chunks as [SOURCE 1], [SOURCE 2], ...
    │
    ▼
build_full_prompt()  — combine system instructions + context + question
    │
    ▼
generate_answer(prompt)  — call Gemini API
    │
    │  OR: format_retrieval_only_response() if no API key
    ▼
Return {answer, mode, chunks, error, model}
    │
    ▼
app.py renders: answer text + expandable source citations
```

## Module Responsibilities

| Module | Responsibility | Imports |
|--------|---------------|---------|
| `document_processing.py` | Load files, extract text, split into chunks | pathlib, fitz (optional) |
| `embeddings.py` | Create and search the vector index | numpy, sentence-transformers |
| `retrieval.py` | Retrieve relevant chunks, build LLM prompt | embeddings.py |
| `llm_provider.py` | Call Gemini Interactions API, handle errors, retrieval-only fallback | google-genai, dotenv |
| `rag_pipeline.py` | Orchestrate all stages | all of the above |
| `quiz.py` | Quiz questions, answer checking, scoring | none |
| `app.py` | Streamlit UI, session state, navigation | rag_pipeline.py, quiz.py |

## Deployment (Streamlit Community Cloud)

1. Push repository to GitHub (ensure .env is in .gitignore).
2. Visit share.streamlit.io → New app → Connect repository.
3. Set main file path: `exam_pilot_ai/app.py`.
4. Add `GEMINI_API_KEY` under App Settings → Secrets.
5. Deploy. The app is publicly accessible via a `.streamlit.app` URL.

Requirements: `requirements.txt` is automatically read by Streamlit Cloud.
