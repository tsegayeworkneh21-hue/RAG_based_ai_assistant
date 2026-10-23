# System Limitations

This document honestly describes what ExamPilot AI cannot do, what risks it carries, and what would be needed to address them.

---

## 1. Hallucination Is Reduced, Not Eliminated

RAG reduces hallucination by requiring the LLM to ground its answer in retrieved evidence. However:
- The LLM may still blend retrieved content with its general training knowledge.
- The system prompt instructs the model to cite only retrieved evidence, but language models do not perfectly follow instructions 100% of the time.
- Answers should always be cross-checked against the original source passages shown in the citation panel.

**Mitigation:** The citation panel is mandatory. Students should read the supporting passages, not just the generated summary.

---

## 2. Scanned PDF Files Are Not Supported

PyMuPDF extracts digital text embedded in a PDF. If a PDF was created by scanning physical pages (photographing pages), the file contains images, not text. PyMuPDF will return empty strings for these pages.

**Detection:** The system warns when a PDF yields no extractable text.

**Mitigation to add:** Integrate Tesseract OCR (via pytesseract) to process image-based PDFs. This requires Tesseract to be installed separately and adds significant complexity.

---

## 3. Coverage Is Limited to the Loaded Materials

The system can only answer questions that are covered in the documents that have been processed and indexed. If a student's actual exam covers a topic not in the uploaded materials, the system will correctly return "insufficient evidence" — but it cannot help with that topic.

**Mitigation:** Students should upload comprehensive course notes for all subjects they expect to be tested on.

---

## 4. Free API Quota Limits

The Gemini free tier has daily request limits. Heavy use during exam preparation may exhaust the quota. When the quota is exceeded, the system falls back to retrieval-only mode automatically.

**Mitigation:** The retrieval-only mode still shows the most relevant passages. Students can read those even without generated explanations.

The application uses the Google GenAI SDK's default request timeout. If Gemini
cannot be reached, it reports a connection error and shows the closest retrieved
course passage verbatim; a natural-language generated explanation requires
working access to the Gemini API.

---

## 5. The Index Is In-Memory and Not Persistent

The vector index is stored in Streamlit session state. It is rebuilt each time the application restarts or the user refreshes. For small corpora (~1000 chunks), rebuilding takes 30–60 seconds.

**Mitigation to add:** Serialize the index (NumPy .npy file) and chunk list (JSON) to disk and reload on startup if the files exist.

---

## 6. No User Authentication

The application has no login system. Anyone with the URL can use it. For a public deployment, this means anyone could send requests to your API key.

**Mitigation:** Streamlit Community Cloud deployments can be made private (requires a Streamlit Teams account). For a student project, keeping the URL private is sufficient.

---

## 7. Chunking May Split Concepts Across Boundaries

Fixed-size character chunking can occasionally split a multi-sentence explanation across two chunks. The 150-character overlap reduces this risk but does not eliminate it. A question whose answer spans exactly the boundary between two chunks might receive a slightly incomplete context.

**Mitigation to add:** Semantic chunking (split on topic shifts) or recursive character chunking following document structure.

---

## 8. Performance Is Not Formally Validated

The evaluation in `docs/EVALUATION.md` reports results on a small manually-created question set (9 questions). This is sufficient to demonstrate the system works, but it is not a statistically rigorous evaluation.

Claims that this system "improves exam results" cannot be made from this evaluation alone. Such a claim would require a controlled study with a treatment group (using the system) and a control group (studying without it), measuring actual exam scores.

---

## 9. Language Support

The embedding model (all-MiniLM-L6-v2) is trained primarily on English text. It will produce lower-quality embeddings for Arabic, French, or other languages. If materials are in another language, a multilingual model (paraphrase-multilingual-MiniLM-L12-v2) would be more appropriate.
