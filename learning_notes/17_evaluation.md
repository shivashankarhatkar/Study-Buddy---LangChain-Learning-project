# 17 · Evaluation: Giving Study Buddy a Report Card

> **In one line:** evaluation means running a fixed **test set** through the app and letting an **LLM judge** grade every answer, which turns "it seems to work" into a **number**.

📄 Code: [step16_evaluation.py](../step16_evaluation.py) · 📋 Test set: [evaluation/test_questions.json](../evaluation/test_questions.json) · ▶️ Run: `python step16_evaluation.py`

---

## 🤔 The problem

You change the chunk size, or the prompt, or the model. **Did the answers get better or worse?** Trying three questions by hand doesn't tell you. A stakeholder wants something like:

> "Study Buddy answered **14 of 15** test questions correctly and made up **nothing**."

## 💡 The idea: an exam with an answer key

```
 ┌──────── test_questions.json (the exam + answer key) ────────┐
 │ Q: Which planet is the largest?   expected: Jupiter...      │
 │ Q: Where does the Calvin cycle happen?  expected: stroma    │
 │ Q: Who wrote Romeo and Juliet?    expected: NOT IN NOTES    │ ← tests honesty
 └──────────────────────────────┬──────────────────────────────┘
                                │ for each question
                                ▼
                    ┌──────────────────────┐
                    │ Study Buddy (step 15)│ ──► answer + passages used
                    └──────────────────────┘
                                │
     expected answer ──────────►│
                                ▼
                    ┌──────────────────────┐
                    │  🧑‍🏫 LLM JUDGE        │ ──► Grade(correct=True, grounded=True,
                    │  (structured output) │           reason="Matches: Jupiter")
                    └──────────────────────┘
                                │ add them all up
                                ▼
          ✅ Correct 14/15   🛡️ Grounded 15/15   🙅 Honest "not found" 3/3
```

## 📏 What we measure

| Metric | Question it answers | Why it matters |
|---|---|---|
| **Correct** | Does the answer mean the same as the expected one? | Usefulness |
| **Grounded** | Is every fact supported by the retrieved passages? | No hallucinations |
| **Honest "not found"** | For questions *not* in the notes, did it refuse? | Trust |

## 🧑‍🏫 The judge = structured output again (step 04)

```python
class Grade(BaseModel):
    correct: bool    # matches the expected answer?
    grounded: bool   # supported by the passages?
    reason: str      # one-sentence explanation

judge = JUDGE_PROMPT | get_chat_model(temperature=0).with_structured_output(Grade)
```

`temperature=0` keeps the judge consistent from run to run.

## 🔭 Two ways to run it

```
 run_evaluation()                      run_langsmith_experiment()
 ─────────────────                     ──────────────────────────
 runs locally                          uploads the test set as a LangSmith DATASET
 saves data/eval_results.json          runs an EXPERIMENT with the same judge
 printed in the terminal   compare experiments over time in LangSmith:
                                       "chunk_size 800 → 14/15,  500 → 12/15"
```

## ⚠️ Good to know

- The test set is written for the **sample notes**. With your own notes, write your own questions.
- Each run makes about **2 LLM calls per question**. On the free tier, set `GEMINI_REQUESTS_PER_MINUTE=10`.
- An LLM judge isn't perfect, so read its `reason` for the failed questions.

## ✅ Remember

- Test set + LLM judge = a repeatable quality score.
- Measure **correctness**, **groundedness** and **honest refusals**.
- Re-run it after every change, so you **know** whether things improved.

⬅️ [16 · Conversational RAG](16_conversational_rag.md) · ➡️ **Next:** [18 · The Web App](18_web_app.md)
