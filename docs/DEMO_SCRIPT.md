# Demonstration Script

Use this script when presenting to your instructor. Follow the steps in order.

---

## Opening Statement (30 seconds)

"This application is called ExamPilot AI. It uses a technique called Retrieval-Augmented Generation — RAG — to help students prepare for their Data Science exit exam.

The key difference from a regular chatbot is that it answers only from specific course materials, and it shows you exactly which passages it used to generate the answer. This makes it honest about what it knows and doesn't know."

---

## Step 1: Show the Knowledge Base Setup (1 minute)

1. Open the app at `http://localhost:8501`.
2. Show the sidebar: "Here I have sample course notes covering Machine Learning, AI, and Statistics."
3. Click **"Build Knowledge Base"**.
4. While it builds: "The system is splitting the documents into 800-character text chunks, then converting each chunk into a 384-dimensional embedding vector using a model called all-MiniLM-L6-v2. These vectors encode the meaning of each passage numerically."
5. Show the success message: "The index is built — [X] chunks from 3 documents."

---

## Step 2: Ask an In-Scope Question (2 minutes)

1. Navigate to the **Q&A** tab.
2. Type: `What is the difference between overfitting and underfitting?`
3. Click **Ask**.
4. Point to the answer: "The system retrieved passages from the course notes, constructed a prompt containing that evidence, and asked Gemini to explain it based only on those passages."
5. **Expand the sources**: "Here you can see exactly which passages were retrieved, with their similarity scores. The system is not inventing these citations."

---

## Step 3: Ask an Out-of-Scope Question (1 minute)

1. Type: `What is the capital of France?`
2. Click **Ask**.
3. Point to the response: "The system says it cannot find sufficient information. It does not hallucinate an answer. The similarity score for the best match was below our threshold of 0.25, so the system correctly recognizes this is outside its knowledge base."

---

## Step 4: Show the Pipeline Inspector (1 minute)

1. Navigate to **Pipeline Inspector**.
2. Show chunk viewer: "This is what a chunk looks like — a 800-character piece of text with its source and page number attached."
3. Type a test query and show raw similarity scores: "You can see the actual cosine similarity scores. Questions about ML get high scores against ML chunks. The French capital question gets low scores against all chunks."

---

## Step 5: Take the Quiz (1 minute)

1. Navigate to **Practice Quiz**.
2. Answer two or three questions.
3. Click **Submit Quiz**.
4. Show the explanations: "Each question has a detailed explanation tied back to the course concepts."

---

## Closing Statement (1 minute)

"In summary:
- This is genuine RAG — real document retrieval, real vector similarity search, real citation.
- The system is honest: it says 'I don't know' rather than guessing.
- All source passages are visible and inspectable.
- The code is modular: document loading, embedding, retrieval, and generation are separate modules that I can explain independently."

---

## Anticipated Questions and Answers

**Q: How is this different from just asking ChatGPT?**  
A: ChatGPT answers from its training data, which may be outdated or wrong. RAG retrieves passages from specific, trusted course materials and grounds the answer in that evidence. The sources are visible and verifiable.

**Q: Can it still hallucinate?**  
A: RAG significantly reduces hallucination, but it does not eliminate it. The LLM is instructed to answer only from retrieved evidence, but language models don't follow instructions perfectly 100% of the time. The citation panel lets you verify each claim.

**Q: What happens when the API key is not configured?**  
A: The system falls back to "retrieval-only mode". It shows the relevant passages without generating a narrative explanation. The core retrieval function still works.

**Q: Why did you choose cosine similarity instead of Euclidean distance?**  
A: Cosine similarity measures the angle between vectors, making it scale-invariant — it doesn't depend on the length of the text. For sentence embeddings, cosine similarity better captures semantic relatedness than Euclidean distance, which is affected by vector magnitude.

**Q: Why all-MiniLM-L6-v2 and not a larger model?**  
A: MiniLM-L6-v2 runs locally on CPU, has excellent performance on semantic similarity benchmarks relative to its size, and produces 384-dimensional vectors that are efficient to store and search. Larger models would be slower to run and require more memory, without proportional quality gains for this task.
