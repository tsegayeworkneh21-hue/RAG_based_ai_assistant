"""
quiz.py
=======
A small practice quiz module grounded in the sample course materials.

DESIGN
------
Questions are stored as plain Python dicts. Each question includes:
  - The question text.
  - Four answer options labelled A through D.
  - The correct answer letter.
  - An explanation citing the relevant concept (not fabricated — drawn from our notes).
  - The subject it belongs to.

The quiz uses session state to track answers and score.
This module is purely data + logic. The Streamlit rendering is in app.py.
"""

import json
import random
import re
import uuid

from src.llm_provider import generate_answer

QUIZ_QUESTIONS = [
    # ---- Machine Learning ----
    {
        "id": "q1",
        "subject": "Machine Learning",
        "question": "A model achieves 98% accuracy on the training set but only 62% on the test set. This is most likely an example of:",
        "options": {
            "A": "Underfitting — the model is too simple.",
            "B": "Overfitting — the model memorised training data and did not generalise.",
            "C": "Good generalisation — 62% test accuracy is expected.",
            "D": "Underfitting — the model has high variance.",
        },
        "answer": "B",
        "explanation": (
            "Overfitting occurs when a model learns training data too well, including its noise. "
            "The large gap between training accuracy (98%) and test accuracy (62%) is the classic sign. "
            "An overfit model has memorised rather than generalised. "
            "The bias-variance trade-off tells us high variance → overfitting."
        ),
    },
    {
        "id": "q2",
        "subject": "Machine Learning",
        "question": "You are building a cancer-detection classifier. Missing a cancer case is far more dangerous than a false alarm. Which metric should you prioritise?",
        "options": {
            "A": "Accuracy.",
            "B": "Precision — minimise false positives.",
            "C": "Recall — minimise false negatives.",
            "D": "AUC — maximise the area under the ROC curve.",
        },
        "answer": "C",
        "explanation": (
            "Recall = TP / (TP + FN). A False Negative means the model missed a real cancer case. "
            "In medical diagnosis, missing a positive (high FN, low Recall) is the most dangerous error. "
            "Prioritising Recall means we accept more false alarms in exchange for catching more real cases."
        ),
    },
    {
        "id": "q3",
        "subject": "Machine Learning",
        "question": "What is the purpose of the test set in a supervised learning workflow?",
        "options": {
            "A": "To train the model parameters.",
            "B": "To tune hyperparameters during training.",
            "C": "To provide an honest estimate of performance on unseen data.",
            "D": "To increase the size of the training set.",
        },
        "answer": "C",
        "explanation": (
            "The test set is held back and never seen during training or hyperparameter tuning. "
            "It provides an unbiased estimate of how the model will perform on truly new data. "
            "Evaluating on training data would give misleadingly high scores because the model may have memorised the examples."
        ),
    },
    {
        "id": "q4",
        "subject": "Machine Learning",
        "question": "Which of the following is an unsupervised learning algorithm?",
        "options": {
            "A": "Logistic Regression.",
            "B": "Random Forest.",
            "C": "K-Means Clustering.",
            "D": "Support Vector Machine.",
        },
        "answer": "C",
        "explanation": (
            "K-Means Clustering is unsupervised — it groups data without any labels. "
            "The algorithm finds structure by minimising the distance between each point and its cluster centre. "
            "Logistic Regression, Random Forest, and SVM are all supervised — they require labelled training data."
        ),
    },
    # ---- Inferential Statistics ----
    {
        "id": "q5",
        "subject": "Inferential Statistics",
        "question": "A study yields a p-value of 0.03 with a significance level α = 0.05. What is the correct conclusion?",
        "options": {
            "A": "Fail to reject H₀ because 0.03 < 0.05.",
            "B": "Reject H₀ because p < α, indicating statistically significant evidence against H₀.",
            "C": "Accept H₁ as definitely true.",
            "D": "The null hypothesis is false with 97% probability.",
        },
        "answer": "B",
        "explanation": (
            "When p < α, we reject H₀. p = 0.03 < α = 0.05 means the observed result would be unlikely "
            "if H₀ were true, so we have statistically significant evidence against H₀. "
            "IMPORTANT: This does NOT mean H₁ is definitely true, nor that H₀ is false with 97% probability. "
            "The p-value is NOT the probability that H₀ is true."
        ),
    },
    {
        "id": "q6",
        "subject": "Inferential Statistics",
        "question": "A researcher constructs a 95% confidence interval for a population mean and gets [42.1, 47.9]. How should this be interpreted?",
        "options": {
            "A": "There is a 95% probability that the true mean is between 42.1 and 47.9.",
            "B": "95% of individual data values fall between 42.1 and 47.9.",
            "C": "If this procedure were repeated many times, 95% of the constructed intervals would contain the true mean.",
            "D": "The sample mean is exactly 95% accurate.",
        },
        "answer": "C",
        "explanation": (
            "The correct frequentist interpretation: if we repeated the sampling procedure many times and "
            "built a CI each time, 95% of those intervals would contain the true population mean. "
            "We cannot say there is a 95% probability the mean is in THIS specific interval — "
            "the true mean is a fixed (unknown) value, not a random variable."
        ),
    },
    {
        "id": "q7",
        "subject": "Inferential Statistics",
        "question": "What does a Type II error (β) represent?",
        "options": {
            "A": "Rejecting H₀ when H₀ is actually true.",
            "B": "Failing to reject H₀ when H₁ is actually true.",
            "C": "The significance level of the test.",
            "D": "The power of the test.",
        },
        "answer": "B",
        "explanation": (
            "Type II error = failing to reject H₀ when H₁ is true (a missed real effect). "
            "Type I error (α) = rejecting H₀ when H₀ is true (a false alarm). "
            "Power = 1 − β = the probability of correctly detecting a real effect. "
            "Increasing sample size increases power without changing α."
        ),
    },
    # ---- Artificial Intelligence ----
    {
        "id": "q8",
        "subject": "Artificial Intelligence",
        "question": "What distinguishes RAG (Retrieval-Augmented Generation) from a standard LLM chatbot?",
        "options": {
            "A": "RAG uses a larger language model.",
            "B": "RAG retrieves relevant passages from a document store and grounds the answer in that evidence.",
            "C": "RAG requires no training data.",
            "D": "RAG answers only yes/no questions.",
        },
        "answer": "B",
        "explanation": (
            "A standard LLM answers from its training data alone and can hallucinate. "
            "RAG first retrieves relevant passages from a trusted document store, then gives those passages "
            "to the LLM as context. The LLM generates its answer based on the retrieved evidence. "
            "This reduces hallucination and allows answering questions about documents the LLM was not trained on."
        ),
    },
    {
        "id": "q9",
        "subject": "Artificial Intelligence",
        "question": "Which activation function is most commonly used in hidden layers of modern neural networks?",
        "options": {
            "A": "Sigmoid.",
            "B": "Softmax.",
            "C": "ReLU (Rectified Linear Unit).",
            "D": "Linear (no activation).",
        },
        "answer": "C",
        "explanation": (
            "ReLU(x) = max(0, x). It is the most widely used activation in hidden layers because: "
            "(1) it is computationally simple; (2) it avoids the vanishing gradient problem that affects "
            "sigmoid and tanh; (3) it allows deep networks to train effectively. "
            "Sigmoid is used in binary output layers. Softmax is used in multi-class output layers."
        ),
    },
    {
        "id": "q10",
        "subject": "Artificial Intelligence",
        "question": "The A* search algorithm is guaranteed to find the optimal (lowest-cost) path when:",
        "options": {
            "A": "The graph has no cycles.",
            "B": "The heuristic function h(n) is admissible — it never overestimates the true cost.",
            "C": "Depth-first search is used as a fallback.",
            "D": "The branching factor is less than 10.",
        },
        "answer": "B",
        "explanation": (
            "A* uses f(n) = g(n) + h(n), where g(n) is the actual cost from start to n, "
            "and h(n) is the heuristic estimate from n to the goal. "
            "If h(n) is admissible (never overestimates), A* is guaranteed to find the optimal path. "
            "A consistent (monotone) heuristic is an even stronger condition that also ensures A* "
            "does not re-expand nodes."
        ),
    },
]


def _extract_course_definitions(chunks: list[dict]) -> list[dict]:
    definitions = []
    seen = set()

    def add_definition(term: str, meaning: str, source: str) -> None:
        term = re.sub(r"\s+", " ", term).strip(" -*\t")
        meaning = re.sub(r"\s+", " ", meaning).strip(" -*\t.,;:")
        key = (term.casefold(), meaning.casefold())
        if len(term) < 2 or len(meaning) < 18 or len(meaning) > 240 or key in seen:
            return
        seen.add(key)
        definitions.append({"term": term, "meaning": meaning, "source": source})

    for chunk in chunks:
        text = str(chunk.get("text", ""))
        source = str(chunk.get("source", "Course material"))
        for line in text.splitlines():
            match = re.match(r"\s*[-*]?\s*([^:]{2,70}):\s*(.+)", line)
            if match:
                add_definition(match.group(1), match.group(2), source)

        flattened = re.sub(r"\s+", " ", text)
        pattern = re.compile(
            r"\b([A-Z][A-Za-z0-9 ()/'-]{2,60}?)\s+"
            r"(?:is|are|refers to|means|measures|describes)\s+"
            r"([^.!?]{18,240})[.!?]"
        )
        for match in pattern.finditer(flattened):
            add_definition(match.group(1), match.group(2), source)

    return definitions


def _generate_local_course_quiz(chunks: list[dict], question_count: int) -> list[dict]:
    definitions = _extract_course_definitions(chunks)
    if len(definitions) < 4:
        return []

    rng = random.Random(uuid.uuid4().int)
    rng.shuffle(definitions)
    quiz_id = uuid.uuid4().hex[:8]
    questions = []

    for fact in definitions:
        distractors = [item["meaning"] for item in definitions if item is not fact]
        if len(distractors) < 3:
            break
        choices = [(fact["meaning"], True)] + [(value, False) for value in distractors[:3]]
        rng.shuffle(choices)
        options = {chr(65 + position): value for position, (value, _) in enumerate(choices)}
        answer = next(chr(65 + position) for position, (_, correct) in enumerate(choices) if correct)
        source = fact["source"]
        subject = source.rsplit(".", 1)[0].replace("_", " ").replace("\\", "/").split("/")[-1].title()
        questions.append({
            "id": f"course_{quiz_id}_{len(questions) + 1}",
            "subject": subject or "Course Material",
            "question": f"According to the course material, what does '{fact['term']}' mean?",
            "options": options,
            "answer": answer,
            "explanation": f"The notes define {fact['term']} as: {fact['meaning']}. Source: {source}.",
        })
        if len(questions) >= question_count:
            break

    return questions


def _course_quiz_fallback(chunks: list[dict], question_count: int, reason: str) -> dict:
    local_questions = _generate_local_course_quiz(chunks, question_count)
    if local_questions:
        return {
            "questions": local_questions,
            "error": None,
            "warning": f"Gemini could not generate a valid quiz ({reason}) — generated a definition-based quiz locally from your course notes.",
        }
    return {
        "questions": [],
        "error": f"{reason} Local quiz generation needs at least four distinct definitions in the course notes.",
    }


def generate_course_quiz(chunks: list[dict], question_count: int = 8) -> dict:
    """Generate validated multiple-choice questions from indexed course chunks."""
    context_parts = []
    remaining_chars = 16000
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if not text or remaining_chars <= 0:
            continue
        source = str(chunk.get("source", "Course material"))
        part = f"[{source}]\n{text[:remaining_chars]}"
        context_parts.append(part)
        remaining_chars -= len(part)

    if not context_parts:
        return {"questions": [], "error": "The loaded course contains no readable text."}

    prompt = f"""Create {question_count} distinct multiple-choice study questions using only the course excerpts below.
Return valid JSON only, with this shape. Keep questions, answer choices, and explanations concise so all {question_count} questions fit in one response:
{{"questions": [{{"subject": "topic", "question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A", "explanation": "..."}}]}}
Make exactly one option correct for each question. Keep every answer and explanation supported by the excerpts. Do not invent facts.

COURSE EXCERPTS:
{chr(10).join(context_parts)}"""

    response = generate_answer(prompt, max_output_tokens=8192)
    if not response.get("success"):
        error = response.get("error") or "Question generation failed."
        return _course_quiz_fallback(chunks, question_count, error)

    raw_answer = (response.get("answer") or "").strip()
    if raw_answer.startswith("```"):
        raw_answer = raw_answer.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(raw_answer)
    except json.JSONDecodeError:
        start, end = raw_answer.find("{"), raw_answer.rfind("}")
        if start < 0 or end <= start:
            return _course_quiz_fallback(
                chunks, question_count, "Gemini returned questions in an invalid format."
            )
        try:
            payload = json.loads(raw_answer[start:end + 1])
        except json.JSONDecodeError:
            return _course_quiz_fallback(
                chunks, question_count, "Gemini returned questions in an invalid format."
            )

    generated = payload.get("questions", []) if isinstance(payload, dict) else []
    questions = []
    quiz_id = uuid.uuid4().hex[:8]
    for item in generated:
        if not isinstance(item, dict):
            continue
        options = item.get("options")
        if not isinstance(options, dict):
            continue
        options = {str(key).strip().upper(): str(value).strip() for key, value in options.items()}
        answer = str(item.get("answer", "")).strip().upper()
        question = str(item.get("question", "")).strip()
        explanation = str(item.get("explanation", "")).strip()
        if not question or not explanation or set(options) != {"A", "B", "C", "D"} or answer not in options:
            continue
        questions.append({
            "id": f"course_{quiz_id}_{len(questions) + 1}",
            "subject": str(item.get("subject") or "Course Material").strip(),
            "question": question,
            "options": options,
            "answer": answer,
            "explanation": explanation,
        })

    if not questions:
        return _course_quiz_fallback(
            chunks, question_count, "Gemini did not return any valid questions."
        )
    return {"questions": questions, "error": None, "warning": None}


def get_questions_by_subject(subject: str = None, questions: list[dict] = None) -> list[dict]:
    """
    Return quiz questions, optionally filtered by subject.

    Parameters
    ----------
    subject : "Machine Learning", "Inferential Statistics", "Artificial Intelligence",
              or None for all questions.
    """
    question_list = QUIZ_QUESTIONS if questions is None else questions
    if subject is None:
        return question_list
    return [q for q in question_list if q["subject"] == subject]


def check_answer(question_id: str, selected_option: str, questions: list[dict] = None) -> dict:
    """
    Check whether a selected answer is correct for a given question.

    Parameters
    ----------
    question_id     : The "id" field of the question dict.
    selected_option : The letter the student selected ("A", "B", "C", or "D").

    Returns
    -------
    A dict with "correct" (bool), "correct_answer" (str), "explanation" (str).
    """
    for q in (QUIZ_QUESTIONS if questions is None else questions):
        if q["id"] == question_id:
            correct = selected_option == q["answer"]
            return {
                "correct": correct,
                "correct_answer": q["answer"],
                "explanation": q["explanation"],
                "selected": selected_option,
            }
    return {"correct": False, "error": f"Question ID {question_id} not found."}


def calculate_score(answers: dict, questions: list[dict] = None) -> dict:
    """
    Calculate quiz score from a dict of {question_id: selected_option}.

    Parameters
    ----------
    answers : Dict mapping question_id to the student's selected option letter.

    Returns
    -------
    A dict with "score" (int), "total" (int), "percentage" (float),
    "correct_ids" (list), "incorrect_ids" (list).
    """
    correct_ids = []
    incorrect_ids = []

    for qid, selected in answers.items():
        result = check_answer(qid, selected, questions)
        if result.get("correct"):
            correct_ids.append(qid)
        else:
            incorrect_ids.append(qid)

    score = len(correct_ids)
    total = len(answers)
    percentage = (score / total * 100) if total > 0 else 0

    return {
        "score": score,
        "total": total,
        "percentage": percentage,
        "correct_ids": correct_ids,
        "incorrect_ids": incorrect_ids,
        "grade": _letter_grade(percentage),
    }


def _letter_grade(percentage: float) -> str:
    """Convert a percentage score to a letter grade."""
    if percentage >= 90:
        return "A"
    elif percentage >= 80:
        return "B"
    elif percentage >= 70:
        return "C"
    elif percentage >= 60:
        return "D"
    else:
        return "F"
