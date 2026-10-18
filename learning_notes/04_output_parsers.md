# 04 · Output Parsers & Structured Output: Clean, Usable Answers

> **In one line:** an output parser turns the model's reply into exactly the **shape your code needs**: a string, a dictionary, or a Python object.

📄 Code: [step04_output_parsers.py](../step04_output_parsers.py)

---

## 🤔 The problem

Study Buddy's quiz feature needs to **show a question, list 4 options, and check your answer**. If Gemini replies with a paragraph of free text like this...

```
"Sure! Here's a quiz. Question 1: Which planet is the largest? You could say Mars,
Jupiter, Earth or Venus... the answer is Jupiter because..."
```

...our code can't reliably find "the options" or "the correct answer" inside it. We need **data**, not prose.

## 💡 The idea: a translator at the end of the line

```
                ┌──────────┐        ┌──────────────────┐
 prompt ──────► │  Gemini  │ ─────► │  Output parser   │ ─────► what your code needs
                └──────────┘ AI     └──────────────────┘
                             Message
```

| You need... | Use... | You get... |
|---|---|---|
| just the text | `StrOutputParser()` | `"Evaporation is when..."` (a `str`) |
| a dictionary | `JsonOutputParser()` | `{"topic": "volcanoes", "key_terms": [...]}` (a `dict`) |
| a typed object ⭐ | `llm.with_structured_output(Quiz)` | `Quiz(topic=..., questions=[...])` |

---

## 1) `StrOutputParser`: AIMessage → string

```
 AIMessage(content="Evaporation is...", usage_metadata=..., ...)
                           │  StrOutputParser
                           ▼
              "Evaporation is..."     (just the text)
```

📍 Used at the end of almost every Study Buddy chain: explain, summary, RAG answer.

## 2) `JsonOutputParser`: JSON text → Python dict

You ask the model for JSON in the prompt, and the parser converts the reply into a `dict`.

```
 '{"topic": "volcanoes", "key_terms": [{"term": "magma", ...}]}'
                           │  JsonOutputParser
                           ▼
 {"topic": "volcanoes", "key_terms": [{"term": "magma", ...}]}   ← now data["key_terms"] works
```

## 3) Structured output with Pydantic ⭐ (the recommended way)

First you **describe the shape** of the data you want using a Pydantic class. This is how Study Buddy defines a quiz:

```python
class QuizQuestion(BaseModel):
    question: str        = Field(description="The question text")
    options: list[str]   = Field(description="Exactly 4 possible answers")
    correct_answers: list[str] = Field(description="ALL correct options (one or more)")
    explanation: str     = Field(description="One sentence explaining why the answer is correct")

class Quiz(BaseModel):
    topic: str
    questions: list[QuizQuestion]
```

Picture it as a form that Gemini **must** fill in:

```
 ┌──────────────── Quiz ────────────────┐
 │ topic: "The planets"                 │
 │ questions:                           │
 │  ┌──────────── QuizQuestion ───────┐ │
 │  │ question: "Largest planet?"     │ │
 │  │ options: [Mars, Jupiter,        │ │
 │  │           Earth, Venus]         │ │
 │  │ correct_answers: ["Jupiter"]    │ │
 │  │ explanation: "It's 11x wider.." │ │
 │  └─────────────────────────────────┘ │
 │  ┌──────────── QuizQuestion ───────┐ │
 │  │ ...                             │ │
 └──────────────────────────────────────┘
```

Then:

```python
quiz_llm = llm.with_structured_output(Quiz)   # a model that ALWAYS replies with a Quiz
quiz = quiz_llm.invoke("Create a 2 question quiz about the water cycle")

quiz.questions[0].options          # → ['Evaporation', 'Condensation', ...]   real Python data!
```

📍 **This powers option 3 in the app (Quiz me):** `main.py` loops over `quiz.questions`, prints the `options`, and compares your choice(s) with `correct_answers`. The web app's Quiz tab shows each option as a checkbox, because a question can have more than one correct answer.

> 💡 The `Field(description=...)` texts are **sent to Gemini as instructions**, so write them clearly!

---

## ⚠️ Common beginner mistakes

| Mistake | Fix |
|---|---|
| Treating the model's reply as a string | add `StrOutputParser()` or use `.text` |
| Asking for JSON in the prompt and hoping for the best | prefer `with_structured_output()`, which is far more reliable |
| Vague field descriptions | describe each field clearly: "Exactly 4 options" |

## ✅ Remember

- Parsers turn the `AIMessage` into what your code needs.
- `StrOutputParser` gives text, `JsonOutputParser` gives a dict, `with_structured_output(Class)` gives an object.
- Study Buddy's quizzes are `Quiz` objects, which is what makes the quiz game possible.

⬅️ [03 · Prompt Templates](03_prompt_templates.md) · ➡️ **Next:** [05 · Chains (LCEL)](05_chains_lcel.md)
