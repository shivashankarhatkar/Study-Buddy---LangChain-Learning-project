"""
STEP 16  -  Evaluation: measuring how GOOD the RAG answers are
==============================================================
📖 Read first: learning_notes/17_evaluation.md  (simple explanation with diagrams)

THE PROBLEM
    "It seems to work" is not enough for a stakeholder. We need a NUMBER:
    "Study Buddy answered 14 of 15 test questions correctly."

THE IDEA: a test set + an LLM judge
    1. A TEST SET: questions with the answer we EXPECT (evaluation/test_questions.json).
       It includes questions that are NOT in the notes - the correct behaviour
       there is to say "I couldn't find that".
    2. Run Study Buddy on every question.
    3. An "LLM-as-a-judge" grades each answer with structured output (step 04):
         correct  -> does the answer match the expected answer?
         grounded -> is every claim supported by the retrieved passages? (no made-up facts)
    4. Add up the scores.

        test question ─► Study Buddy (step 15) ─► answer + passages
                                                     │
              expected answer ──────────────────────►├─► LLM judge ─► {correct, grounded, reason}
                                                     ▼
                                              summary: 14/15 correct

TWO WAYS TO RUN IT
    run_evaluation()           -> runs locally, prints the score and saves it to data/eval_results.json
    run_langsmith_experiment() -> uploads the test set to LangSmith as a DATASET and runs an
                                  EXPERIMENT, so you can compare results over time in the UI

Run this file:   python step16_evaluation.py
(Note: ~30 LLM calls. On the free Gemini tier set GEMINI_REQUESTS_PER_MINUTE=10 in .env.)
"""

import json
import os
from datetime import datetime

from pydantic import BaseModel, Field
from langsmith import Client, traceable
from langchain_core.prompts import ChatPromptTemplate

from config import get_chat_model, EVAL_DATASET_PATH, EVAL_RESULTS_PATH
from step10_vector_store import load_vector_store
from step15_conversational_rag import ask, format_numbered_passages


LANGSMITH_DATASET_NAME = "study-buddy-test-questions"


# ------------------------------------------------------------------
# The judge: structured output (step 04) used for grading
# ------------------------------------------------------------------
class Grade(BaseModel):
    """The judge's verdict on one answer."""
    correct: bool = Field(description="True if the answer means the same as the expected answer. "
                                      "If the expected answer says it is not in the notes, True only if "
                                      "the assistant said it couldn't find it.")
    grounded: bool = Field(description="True if every fact in the answer is supported by the passages "
                                       "(a 'couldn't find it' answer counts as grounded).")
    reason: str = Field(description="One short sentence explaining the verdict")


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict teacher grading a study assistant's answer. "
     "Compare it with the expected answer and check it against the passages it was given."),
    ("human",
     "QUESTION:\n{question}\n\nEXPECTED ANSWER:\n{expected}\n\n"
     "ASSISTANT'S ANSWER:\n{answer}\n\nPASSAGES THE ASSISTANT USED:\n{passages}"),
])


def grade_answer(question: str, expected: str, answer: str, passages_text: str) -> Grade:
    judge = JUDGE_PROMPT | get_chat_model(temperature=0).with_structured_output(Grade)
    return judge.invoke(
        {"question": question, "expected": expected, "answer": answer, "passages": passages_text or "(none)"},
        config={"run_name": "Judge Answer"},
    )


def load_test_set() -> list[dict]:
    with open(EVAL_DATASET_PATH, encoding="utf-8") as file:
        return json.load(file)


# ------------------------------------------------------------------
# Way 1: local evaluation (used by the web app)
# ------------------------------------------------------------------
@traceable(name="Quality Evaluation", process_inputs=lambda inputs: {})
def run_evaluation(on_progress=None) -> dict:
    """
    Grade every test question and save the results.
    on_progress(done, total) is an optional function the web app uses to move its progress bar.
    """
    vector_store = load_vector_store()
    test_set = load_test_set()
    rows = []

    for number, item in enumerate(test_set, start=1):
        result = ask(vector_store, item["question"])
        grade = grade_answer(item["question"], item["reference_answer"], result["answer"],
                             format_numbered_passages(result["passages"]))
        rows.append({
            "question": item["question"],
            "expected": item["reference_answer"],
            "answer": result["answer"],
            "in_notes": item["in_notes"],
            "correct": grade.correct,
            "grounded": grade.grounded,
            "reason": grade.reason,
            "passages_used": len(result["passages"]),
        })
        if on_progress:
            on_progress(number, len(test_set))

    total = len(rows)
    summary = {
        "total": total,
        "correct": sum(r["correct"] for r in rows),
        "grounded": sum(r["grounded"] for r in rows),
        "refusals_correct": sum(r["correct"] for r in rows if not r["in_notes"]),
        "refusals_total": sum(1 for r in rows if not r["in_notes"]),
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    results = {"summary": summary, "rows": rows}

    os.makedirs(os.path.dirname(EVAL_RESULTS_PATH), exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    return results


def load_last_results() -> dict | None:
    """Return the most recent saved evaluation, or None if it was never run."""
    if not os.path.exists(EVAL_RESULTS_PATH):
        return None
    with open(EVAL_RESULTS_PATH, encoding="utf-8") as file:
        return json.load(file)


# ------------------------------------------------------------------
# Way 2: a LangSmith experiment (needs LANGSMITH_API_KEY)
# ------------------------------------------------------------------
def run_langsmith_experiment() -> str:
    """Upload the test set as a LangSmith dataset (once) and run an experiment on it."""
    client = Client()

    # 1. Create the dataset the first time. Each example = inputs + expected outputs.
    if not client.has_dataset(dataset_name=LANGSMITH_DATASET_NAME):
        dataset = client.create_dataset(dataset_name=LANGSMITH_DATASET_NAME,
                                        description="Study Buddy RAG test questions")
        client.create_examples(dataset_id=dataset.id, examples=[
            {"inputs": {"question": item["question"]},
             "outputs": {"answer": item["reference_answer"], "in_notes": item["in_notes"]}}
            for item in load_test_set()
        ])

    vector_store = load_vector_store()

    # 2. The "target": the thing being tested. Input = example inputs, output = what our app says.
    def study_buddy(inputs: dict) -> dict:
        result = ask(vector_store, inputs["question"])
        return {"answer": result["answer"], "passages": format_numbered_passages(result["passages"])}

    # 3. The evaluator: LangSmith calls it for every example and stores the scores.
    def judge(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        grade = grade_answer(inputs["question"], reference_outputs["answer"], outputs["answer"], outputs["passages"])
        return {"results": [
            {"key": "correct", "score": int(grade.correct), "comment": grade.reason},
            {"key": "grounded", "score": int(grade.grounded)},
        ]}

    # 4. Run it. max_concurrency=1 keeps us inside free-tier rate limits.
    results = client.evaluate(study_buddy, data=LANGSMITH_DATASET_NAME, evaluators=[judge],
                              experiment_prefix="study-buddy-rag", max_concurrency=1)
    return results.experiment_name


if __name__ == "__main__":
    def show_progress(done, total):
        print(f"  graded {done}/{total}")

    print("=== Running the evaluation ===")
    results = run_evaluation(on_progress=show_progress)
    s = results["summary"]
    print(f"\nCorrect : {s['correct']}/{s['total']}")
    print(f"Grounded: {s['grounded']}/{s['total']}")
    print(f"Correct 'not found' answers: {s['refusals_correct']}/{s['refusals_total']}")

    print("\nQuestions that failed:")
    for row in results["rows"]:
        if not row["correct"]:
            print(f"  ✗ {row['question']}\n    → {row['reason']}")

    if os.getenv("LANGSMITH_API_KEY"):
        print("\n=== Running a LangSmith experiment ===")
        print("Experiment:", run_langsmith_experiment(), "(open Datasets & Experiments in LangSmith)")
