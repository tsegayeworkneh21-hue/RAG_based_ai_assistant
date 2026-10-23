# ExamPilot AI 🎓
Repository: `RAG_based_ai_assistant`
## Data Science Exit Exam Preparation Assistant Using Retrieval-Augmented Generation (RAG)

> **Academic Project** — Fourth-Year Data Science Capstone  
> *Sample educational materials only. Not official university examination content.*

---

## What This Application Does

ExamPilot AI is an AI-powered study assistant that helps Data Science students prepare for their exit examination. When a student asks a question, the system:

1. Searches course materials **semantically** (by meaning, not just keywords)
2. Retrieves the most relevant passages as **evidence**
3. Passes that evidence to a language model to generate a **grounded explanation**
4. Shows the **exact sources and passages** that supported the answer
5. Returns a clear **"insufficient evidence"** message when the materials don't cover the question

This is genuine RAG — not a chatbot that answers from general knowledge.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key (optional — the app works without it in retrieval-only mode)

```bash
# Copy the example file
copy .env.example .env
# Edit .env and add your Gemini API key
# Get a free key at: https://aistudio.google.com/
```

### 3. Run the application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

If you already installed the project before the Gemini SDK update, reinstall the
requirements with the same Python environment before starting the app:

```powershell
python -m pip install -r requirements.txt
```

### If asking a question takes too long

The app does not set a custom request timeout; it uses the Google GenAI SDK's
default timeout. If Gemini cannot be reached, the app reports a connection
error and shows the closest course passage verbatim instead of an AI-generated
explanation. Check internet access and whether a VPN, proxy, or firewall blocks
Google API connections. Verify the `GEMINI_API_KEY` in `.env` if the connection
works but generation still fails.

### Troubleshooting vector-index builds on Windows

If building the knowledge base fails with a PyTorch `c10.dll` or `WinError 126`
message, install or repair the **Microsoft Visual C++ 2015–2022 Redistributable
(x64)** from [Microsoft's official download page](https://aka.ms/vs/17/release/vc_redist.x64.exe).
Restart the terminal, then check that PyTorch imports in the same Python
environment used by Streamlit:

```powershell
python -c "import torch; print(torch.__version__)"
```

If that command reports that `sentence_transformers` is missing, install the
project requirements using that same interpreter:

```powershell
python -m pip install -r requirements.txt
```

Then start the app with `python -m streamlit run app.py`. The app displays the
underlying embedding error if index creation fails.

---

## Project Structure

```
exam_pilot_ai/
├── app.py                          # Streamlit user interface
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── LEARNING_GUIDE.md               # Concepts and code walkthroughs
├── .env.example                    # API key template
├── .gitignore                      # Files excluded from Git
│
├── src/                            # Core pipeline modules
│   ├── config.py                   # Validated pipeline settings
│   ├── document_processing.py      # STEPS 1–3: Load, extract, chunk
│   ├── embeddings.py               # STEPS 4–5: Embed and index
│   ├── retrieval.py                # STEPS 6–7: Retrieve and build prompt
│   ├── llm_provider.py             # STEP 8:   Generate answer
│   ├── rag_pipeline.py             # Orchestrator: connects all steps
│   ├── evaluation.py               # Verified QA retrieval evaluation
│   └── quiz.py                     # Practice quiz questions and scoring
│
├── data/
│   └── sample_materials/           # Original course notes (3 subjects)
│       ├── machine_learning_notes.txt
│       ├── ai_fundamentals_notes.txt
│       ├── inferential_statistics_notes.txt
│       └── verified_qa_set.json    # Test questions with expected answers
│
├── tests/                          # Automated tests
│   ├── test_document_processing.py
│   ├── test_config.py
│   ├── test_evaluation.py
│   └── test_retrieval.py
│
└── docs/                           # Documentation
    ├── ARCHITECTURE.md
    ├── DECISION_LOG.md
    ├── EVALUATION.md
    ├── DEMO_SCRIPT.md
    └── LIMITATIONS.md
```

---

## Running Tests

```bash
# From the exam_pilot_ai directory
pytest tests/ -v

# Run a specific test file
pytest tests/test_document_processing.py -v

# Retrieval evaluation against verified_qa_set.json (needs sentence-transformers)
python -m src.evaluation
```

Tests that require the embedding model are automatically skipped if `sentence-transformers` is not installed.

`src/evaluation.py` is the production form of the exploratory QA analysis: it validates the labelled JSON, runs retrieval with `PipelineConfig`, and reports hit rate, threshold accuracy, out-of-scope rejection, and citation validity. See `docs/EVALUATION.md`.

---

## Subjects Covered

- **Machine Learning**: supervised/unsupervised learning, classification, regression, overfitting, metrics (precision, recall, F1)
- **Introduction to Artificial Intelligence**: search algorithms, neural networks, deep learning, NLP, RAG
- **Inferential Statistics**: hypothesis testing, p-values, confidence intervals, Type I/II errors

---

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| UI | Streamlit | Python-native, no frontend code needed |
| PDF extraction | PyMuPDF | Fast, reliable, preserves page numbers |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Local, fast, high-quality semantic embeddings |
| Vector search | NumPy cosine similarity | Simple, no extra database, sufficient for this scale |
| LLM | Google GenAI SDK — Gemini 3.8 Flash Interactions API | Current supported API |
| Config | python-dotenv | Keeps API keys out of source code |
| Tests | pytest | Standard Python testing framework |

---

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repository.
4. Add `GEMINI_API_KEY` in the app's Secrets settings.
5. Deploy.

See `docs/ARCHITECTURE.md` for the full deployment guide.

---

## Limitations

- The application answers only from the provided course materials.
- Scanned PDF files (image-based) require OCR — not currently implemented.
- The free Gemini tier has daily request limits.
- The system reduces hallucination but does not eliminate it completely.

See `docs/LIMITATIONS.md` for a full discussion.

---

## Academic Use

This project was built as a senior capstone demonstrating practical application of RAG for educational question answering. All sample materials are original. No official exam questions are included or represented.
