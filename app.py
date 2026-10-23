"""
app.py
======
Main Streamlit application for the Data Science Exit Exam Preparation Assistant.

HOW TO RUN
----------
From the exam_pilot_ai directory:
    streamlit run app.py

Or from the project root:
    streamlit run exam_pilot_ai/app.py

ARCHITECTURE OVERVIEW
---------------------
This file is the user interface only. All logic lives in the src/ modules:
  - document_processing.py : Load and chunk documents.
  - embeddings.py          : Create and search vector embeddings.
  - retrieval.py           : Retrieve evidence and build LLM prompts.
  - llm_provider.py        : Call Gemini API or fall back gracefully.
  - rag_pipeline.py        : Orchestrate the above into one clean interface.
  - quiz.py                : Practice quiz questions and scoring.

Streamlit re-runs the entire script every time the user interacts with the UI.
We use st.session_state to preserve data (the loaded index, answers, etc.) across re-runs.
"""

import streamlit as st
import os
import sys
from pathlib import Path

# ---- Make src/ importable regardless of where streamlit is run from ----
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import DEFAULT_CONFIG
from src.document_processing import load_all_from_directory, load_uploaded_file, chunk_pages
from src.embeddings import get_embedding_error
from src.rag_pipeline import build_knowledge_base, answer_question, get_pipeline_status
from src.llm_provider import is_api_available
from src.quiz import (
    get_questions_by_subject,
    check_answer,
    calculate_score,
    generate_course_quiz,
    QUIZ_QUESTIONS,
)

# ---- Page Configuration ----
st.set_page_config(
    page_title="ExamPilot AI — DS Exit Exam Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Sample materials path ----
SAMPLE_DIR = ROOT_DIR / "data" / "sample_materials"


# ===========================================================================
# Session State Initialisation
# ===========================================================================
# Streamlit re-runs from top on every interaction. We store persistent objects
# in st.session_state so they survive re-runs.

def init_session_state():
    """Initialise session state variables on first run."""
    if "index" not in st.session_state:
        st.session_state.index = None          # VectorIndex object
    if "chunks_loaded" not in st.session_state:
        st.session_state.chunks_loaded = False
    if "num_chunks" not in st.session_state:
        st.session_state.num_chunks = 0
    if "sources_loaded" not in st.session_state:
        st.session_state.sources_loaded = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []     # List of {question, response}
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}     # {question_id: selected_option}
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    if "quiz_subject" not in st.session_state:
        st.session_state.quiz_subject = "All Subjects"
    if "course_quiz_questions" not in st.session_state:
        st.session_state.course_quiz_questions = []
    if "quiz_generation" not in st.session_state:
        st.session_state.quiz_generation = 0
    if "quiz_generation_warning" not in st.session_state:
        st.session_state.quiz_generation_warning = None


init_session_state()


# ===========================================================================
# Sidebar
# ===========================================================================

def render_sidebar():
    """Render the sidebar with document loading and pipeline status."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/graduation-cap.png", width=60)
        st.title("ExamPilot AI")
        st.caption("DS Exit Exam Preparation Assistant")
        st.divider()

        # ---- Knowledge Base Setup ----
        st.subheader("📚 Knowledge Base")

        load_option = st.radio(
            "Load materials from:",
            ["Sample Course Materials", "Upload My Own Files"],
            key="load_option",
        )

        uploaded_files = None
        if load_option == "Upload My Own Files":
            uploaded_files = st.file_uploader(
                "Upload PDF or TXT files",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                help="Upload your own course notes in PDF or plain text format.",
            )

        if st.button("⚡ Build Knowledge Base", type="primary", use_container_width=True):
            build_index(load_option, uploaded_files)

        # ---- Pipeline Status ----
        st.divider()
        st.subheader("🔧 Pipeline Status")

        if st.session_state.index and st.session_state.chunks_loaded:
            st.success(f"✅ Index ready — {st.session_state.num_chunks} chunks")
            for src in st.session_state.sources_loaded:
                st.caption(f"  📄 {src}")
        else:
            st.warning("⚠️ No index built yet. Click 'Build Knowledge Base'.")

        # ---- API Status ----
        st.divider()
        if is_api_available():
              st.info("🤖 LLM: Gemini API key configured (connection not tested)")
        else:
            st.warning("📋 LLM: Retrieval-only mode\n\nAdd GEMINI_API_KEY to .env for full RAG")

        # ---- Settings ----
        st.divider()
        st.subheader("⚙️ Settings")
        st.session_state.top_k = st.slider(
            "Retrieved passages (top-k)", min_value=1, max_value=10, value=DEFAULT_CONFIG.top_k,
            help="How many passages to retrieve for each question."
        )
        st.session_state.min_score = st.slider(
            "Min. relevance threshold", min_value=0.1, max_value=0.6, value=DEFAULT_CONFIG.min_score, step=0.05,
            help="Chunks below this similarity score are considered irrelevant."
        )


def build_index(load_option: str, uploaded_files):
    """Load documents, chunk them, and build the vector index."""
    with st.spinner("Loading and processing documents..."):
        all_chunks = []

        if load_option == "Sample Course Materials":
            if not SAMPLE_DIR.exists():
                st.error(f"Sample materials directory not found: {SAMPLE_DIR}")
                return
            all_chunks = load_all_from_directory(str(SAMPLE_DIR))

        elif load_option == "Upload My Own Files" and uploaded_files:
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.read()
                pages = load_uploaded_file(file_bytes, uploaded_file.name)
                chunks = chunk_pages(pages)
                all_chunks.extend(chunks)
                st.caption(f"Loaded: {uploaded_file.name} → {len(chunks)} chunks")
        else:
            st.warning("Please select or upload files before building the index.")
            return

        if not all_chunks:
            st.error("No content could be extracted from the documents.")
            return

    with st.spinner(f"Building vector index for {len(all_chunks)} chunks (embedding model loading...)"):
        index = build_knowledge_base(all_chunks)

    if index.is_built:
        st.session_state.index = index
        st.session_state.chunks_loaded = True
        st.session_state.course_quiz_questions = []
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False
        st.session_state.quiz_generation_warning = None
        st.session_state.quiz_generation += 1
        st.session_state.num_chunks = len(all_chunks)
        stats = index.get_stats()
        st.session_state.sources_loaded = stats.get("sources", [])
        st.success(f"✅ Knowledge base built: {len(all_chunks)} chunks from {len(stats.get('sources', []))} source(s).")
        st.rerun()
    else:
        error = get_embedding_error()
        message = "Failed to build the vector index."
        if error:
            message += f" Embedding error: {error}"
        st.error(message)
        if error and ("c10.dll" in error.lower() or "winerror 126" in error.lower()):
            st.info(
                "On Windows, install or repair the Microsoft Visual C++ 2015–2022 "
                "Redistributable (x64), then reopen your terminal and verify PyTorch "
                "with `python -c \"import torch; print(torch.__version__)\"`. "
                "Run Streamlit with the same Python interpreter."
            )
        elif error and "sentence-transformers is not installed" in error.lower():
            st.info(
                "Install the dependency into the same Python environment used to run "
                "Streamlit: `python -m pip install -r requirements.txt`."
            )


# ===========================================================================
# Main Pages
# ===========================================================================

def render_home():
    """Landing page with project overview."""
    st.title("🎓 Data Science Exit Exam Preparation Assistant")
    st.markdown(
        """
        **An AI-powered study assistant using Retrieval-Augmented Generation (RAG)**

        This application helps you prepare for your Data Science exit examination by:
        - 📖 Searching your course materials semantically (not just by keyword)
        - 🤖 Generating explanations grounded in your actual notes
        - 🔍 Showing you exactly which passages support each answer
        - ❓ Offering practice quiz questions with detailed explanations

        ---
        ### How to use:
        1. **Build the knowledge base** using the sidebar (sample or your own materials)
        2. **Ask a question** in the Q&A tab
        3. **Review sources** to see which passages supported the answer
        4. **Take the quiz** to test your understanding

        > *This is a prototype for academic purposes. Materials are original sample notes,
        > not official university examination documents.*
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Subjects Covered**\n\n• Machine Learning\n• Artificial Intelligence\n• Inferential Statistics")
    with col2:
        st.info("**RAG Pipeline**\n\n• PDF & TXT ingestion\n• Semantic chunking\n• Vector similarity search")
    with col3:
        st.info("**Honest AI**\n\n• Cites exact sources\n• Shows relevance scores\n• Admits uncertainty")


def render_qa():
    """Main Q&A interface."""
    st.header("💬 Ask a Question")

    if not st.session_state.chunks_loaded:
        st.warning("Please build the knowledge base first using the sidebar.")
        return

    # ---- Question Input ----
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g. What is the difference between overfitting and underfitting?",
        key="question_input",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_btn = st.button("Ask", type="primary")
    with col2:
        if st.button("Clear history"):
            st.session_state.chat_history = []
            st.rerun()

    if ask_btn and query.strip():
        with st.spinner("Searching course materials and generating answer..."):
            response = answer_question(
                query=query.strip(),
                index=st.session_state.index,
                top_k=st.session_state.get("top_k", DEFAULT_CONFIG.top_k),
                min_score=st.session_state.get("min_score", DEFAULT_CONFIG.min_score),
            )
        # Add to history (newest first)
        st.session_state.chat_history.insert(0, {
            "question": query.strip(),
            "response": response,
        })

    # ---- Display Chat History ----
    for entry in st.session_state.chat_history:
        q = entry["question"]
        r = entry["response"]

        st.markdown(f"**🙋 Question:** {q}")

        mode = r.get("mode", "unknown")

        if mode == "generated":
            st.success("**🤖 AI Answer** *(Generated using Gemini + retrieved evidence)*")
            st.markdown(r["answer"])

        elif mode == "retrieval_only":
            if r.get("error"):
                st.warning(f"⚠️ {r['error']}")
            st.markdown(r["answer"])

        elif mode == "insufficient_evidence":
            st.error("**📭 Insufficient Evidence**")
            st.markdown(r["answer"])

        elif mode == "error":
            st.error(f"**❌ Error:** {r.get('error', 'Unknown error')}")
            st.markdown(r.get("answer", ""))

        # ---- Source Citations ----
        chunks = r.get("chunks", [])
        if chunks:
            with st.expander(f"📚 View {len(chunks)} source passage(s)", expanded=False):
                for i, chunk in enumerate(chunks, start=1):
                    source = chunk.get("source", "Unknown")
                    page = chunk.get("page")
                    score = chunk.get("score", 0)
                    text = chunk.get("text", "")

                    page_str = f" | Page {page}" if page else ""
                    st.markdown(f"**[SOURCE {i}]** `{source}`{page_str} — Relevance: `{score:.3f}`")
                    st.text_area(
                        label="",
                        value=text[:600] + ("..." if len(text) > 600 else ""),
                        height=120,
                        disabled=True,
                        key=f"chunk_{id(chunk)}_{i}",
                    )

        st.divider()


def render_quiz():
    """Practice quiz interface."""
    st.header("📝 Practice Quiz")
    has_course = st.session_state.index is not None and st.session_state.chunks_loaded
    if has_course:
        st.caption("Generate questions grounded in the course currently loaded in the knowledge base.")
        if st.button("🎲 Generate Quiz from Loaded Course", type="primary"):
            with st.spinner("Creating questions from your course materials..."):
                result = generate_course_quiz(st.session_state.index.chunks)
            if result["error"]:
                st.session_state.quiz_generation_warning = None
                st.error(result["error"])
            else:
                st.session_state.course_quiz_questions = result["questions"]
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_generation_warning = result.get("warning")
                st.session_state.quiz_generation += 1
                st.rerun()

    if st.session_state.quiz_generation_warning:
        st.warning(st.session_state.quiz_generation_warning)

    active_questions = st.session_state.course_quiz_questions or QUIZ_QUESTIONS
    if st.session_state.course_quiz_questions:
        st.caption("Questions are generated from the currently loaded course. Generate again for a fresh set.")
    else:
        st.caption("Showing the built-in sample quiz. Load course materials and generate a course-specific quiz above.")

    # ---- Subject Filter ----
    subjects = ["All Subjects"] + sorted({question["subject"] for question in active_questions})
    selected_subject = st.selectbox(
        "Filter by subject:",
        subjects,
        key=f"quiz_subject_select_{st.session_state.quiz_generation}",
    )

    filter_subject = None if selected_subject == "All Subjects" else selected_subject
    questions = get_questions_by_subject(filter_subject, active_questions)

    if st.button("🔄 Reset Quiz", type="secondary"):
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False
        st.session_state.quiz_generation += 1
        st.rerun()

    # ---- Render Questions ----
    for q in questions:
        st.markdown(f"**{q['id'].upper()}. [{q['subject']}]** {q['question']}")

        options = list(q["options"].items())  # [(letter, text), ...]
        option_labels = [f"{letter}. {text}" for letter, text in options]

        # Find the current selection (if any)
        current_answer = st.session_state.quiz_answers.get(q["id"])
        current_index = None
        if current_answer:
            for i, (letter, _) in enumerate(options):
                if letter == current_answer:
                    current_index = i
                    break

        selected_label = st.radio(
            label="",
            options=option_labels,
            index=current_index,
            key=f"quiz_{st.session_state.quiz_generation}_{q['id']}",
            disabled=st.session_state.quiz_submitted,
        )

        # Parse the letter from the selected label
        if selected_label:
            selected_letter = selected_label.split(".")[0].strip()
            st.session_state.quiz_answers[q["id"]] = selected_letter

        # Show explanation if quiz has been submitted
        if st.session_state.quiz_submitted:
            result = check_answer(
                q["id"], st.session_state.quiz_answers.get(q["id"], ""), active_questions
            )
            if result["correct"]:
                st.success(f"✅ Correct! {result['explanation']}")
            else:
                st.error(
                    f"❌ Incorrect. Correct answer: **{result['correct_answer']}**\n\n{result['explanation']}"
                )

        st.divider()

    # ---- Submit / Score ----
    if not st.session_state.quiz_submitted:
        answered = sum(q["id"] in st.session_state.quiz_answers for q in questions)
        total = len(questions)
        st.caption(f"Answered: {answered} / {total}")

        if st.button("Submit Quiz", type="primary", disabled=(answered < total)):
            st.session_state.quiz_submitted = True
            st.rerun()
        if answered < total:
            st.caption("Answer all questions to submit.")
    else:
        # Show score
        visible_answers = {
            q["id"]: st.session_state.quiz_answers[q["id"]]
            for q in questions
            if q["id"] in st.session_state.quiz_answers
        }
        score_result = calculate_score(visible_answers, active_questions)
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score", f"{score_result['score']} / {score_result['total']}")
        with col2:
            st.metric("Percentage", f"{score_result['percentage']:.1f}%")
        with col3:
            st.metric("Grade", score_result["grade"])


def render_pipeline_debug():
    """Show pipeline internals for learning and debugging."""
    st.header("🔬 Pipeline Inspector")
    st.caption("Use this view to understand what the RAG system is doing under the hood.")

    if not st.session_state.chunks_loaded:
        st.warning("Build the knowledge base first.")
        return

    index = st.session_state.index
    stats = index.get_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Chunks indexed", stats["num_chunks"])
        st.metric("Vector dimensions", stats["vector_dimensions"])
    with col2:
        st.metric("Sources loaded", len(stats.get("sources", [])))
        from src.llm_provider import is_api_available
        st.metric("LLM available", "Yes" if is_api_available() else "No (retrieval-only)")

    st.subheader("Sample Chunk Viewer")
    st.caption("See what a chunk looks like after document processing and chunking.")

    chunk_num = st.number_input(
        "View chunk number:", min_value=0,
        max_value=max(0, stats["num_chunks"] - 1), value=0, step=1
    )
    if index.chunks:
        chunk = index.chunks[chunk_num]
        st.markdown(f"**Source:** `{chunk['source']}` | **Page:** {chunk.get('page', 'N/A')} | **Index:** {chunk['chunk_index']}")
        st.text_area("Chunk text:", value=chunk["text"], height=200, disabled=True)

    st.subheader("Retrieval Inspector")
    st.caption("Test the retrieval step in isolation — see raw similarity scores.")

    test_query = st.text_input("Test query:", placeholder="Type a question to see raw retrieval results...")
    if test_query:
        from src.retrieval import retrieve
        result = retrieve(test_query, index, top_k=5, min_score=0.0)  # min_score=0 to see all
        st.markdown(f"**Found {len(result['all_results'])} results.** Threshold is 0.25.")
        for i, chunk in enumerate(result["all_results"], 1):
            score = chunk.get("score", 0)
            color = "green" if score >= 0.25 else "red"
            st.markdown(
                f"**[{i}]** `{chunk['source']}` | Score: "
                f":{color}[{score:.4f}] {'✅' if score >= 0.25 else '❌ below threshold'}"
            )
            with st.expander(f"Preview chunk {i}"):
                st.write(chunk["text"][:400])


# ===========================================================================
# Main Navigation
# ===========================================================================

render_sidebar()

tab_home, tab_qa, tab_quiz, tab_debug = st.tabs([
    "🏠 Home", "💬 Q&A", "📝 Practice Quiz", "🔬 Pipeline Inspector"
])

with tab_home:
    render_home()

with tab_qa:
    render_qa()

with tab_quiz:
    render_quiz()

with tab_debug:
    render_pipeline_debug()
