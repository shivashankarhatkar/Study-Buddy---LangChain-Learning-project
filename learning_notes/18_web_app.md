# 18 · The Web App: Putting a Face on Study Buddy

> **In one line:** the Streamlit app is only a **frontend** with three pages. Every bit of LangChain logic still lives in the step files, and the app simply calls them.

📄 Code: [app/](../app/) · Run: `streamlit run app/streamlit_app.py` → opens http://localhost:8501

---

## 🧱 Frontend vs. LangChain logic

```
 ┌──────────────────── app/ (Streamlit = what you SEE) ────────────────────┐
 │  streamlit_app.py (top menu)    shared.py (vector store + settings)     │
 │  views/ask.py   ·   views/study.py   ·   views/settings.py              │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │  imports and calls
 ┌──────────────────── step files (LangChain = how it THINKS) ─────────────┐
 │ step05 quiz & summary chains · step07 loader · step10 vector store      │
 │ step14 tracing + feedback · step15 conversational RAG                   │
 └─────────────────────────────────────────────────────────────────────────┘
```

So if you understand the step files, you already understand the app.

## 🗺️ The three pages

| Page | What it calls | Concepts |
|---|---|---|
| 💬 **Ask** | `prepare_answer()` → `build_answer_chain().stream()` | Conversational RAG, memory, streaming, citations, tracing, feedback |
| 🎯 **Quiz & Summary** | `retrieve_passages()` → `build_quiz_chain()` · `load_file()` → `build_summary_chain().stream()` | RAG, structured output, prompt `.partial()`, streaming |
| ⚙️ **Settings** | `add_file_to_store()`, `remove_source_from_store()`, `list_sources()` | Loader → splitter → embeddings → vector store |

## ⚙️ How a setting travels to LangChain

```
 Settings page                     st.session_state           Ask / Quiz & Summary page
 ┌───────────────────────┐        ┌──────────────────┐        ┌───────────────────────────────────┐
 │ slider key="k"        │ ─────► │ k = 6            │ ─────► │ prepare_answer(..., k=6)          │
 │ slider "temperature"  │ ─────► │ temperature = 0.7│ ─────► │ build_answer_chain(temperature=.7)│
 │ "quiz_difficulty"     │ ─────► │ "Hard"           │ ─────► │ quiz chain: {"difficulty": "hard"}│
 └───────────────────────┘        └──────────────────┘        └───────────────────────────────────┘
```

The quiz prompt uses `.partial(difficulty="medium")` to give `{difficulty}` a **default value**. The web app overrides it with the chosen level; older callers that don't pass it simply get "medium".

## 💬 The Ask page: a growing conversation

```
 ┌──────────────────────────────────────────────┐
 │ 🧑‍🎓 Tell me about Jupiter                      │ ← stays on screen
 │ 📚 Jupiter is the largest planet [1]...       │
 │    ▸ 🔍 Under the hood        👍 👎           │ ← hidden until clicked
 │ 🧑‍🎓 How many moons does it have?               │
 │ 📚 Jupiter has at least 95 moons [1].         │
 │    ▸ 🔍 Under the hood        👍 👎           │
 ├──────────────────────────────────────────────┤
 │ [ Ask a question about your notes…      ➤ ]  │ ← always pinned to the bottom
 └──────────────────────────────────────────────┘
```

- The chat list lives in `st.session_state.chat`, so old messages are never erased.
- The same list becomes the **history** (`to_history(chat)`) that makes follow-up questions work (step 15).

## 🎯 Quiz: one or more correct answers

The `QuizQuestion` schema (step 04) has `correct_answers: list[str]`, so a question can have several right answers. Each option is a checkbox, and a question counts as correct only if you tick **exactly** the right options:

```python
def answers_match(chosen, correct):
    return {c.strip().lower() for c in chosen} == {c.strip().lower() for c in correct}
```

## 🔄 Two Streamlit ideas worth knowing

1. **The script re-runs on every click.** Streamlit runs the page file from top to bottom each time you interact with it.
   - `st.session_state` keeps things between re-runs: the chat, the quizzes, the summaries, the settings.
   - `@st.cache_resource` loads the vector store **once** and shares it between pages.
2. **Streaming:** `st.write_stream(generator)` prints text as it arrives. We give it the chunks from `chain.stream(...)`.

## ✅ Remember

- `app/` = the frontend; the step files = the LangChain logic.
- Settings → `st.session_state` → function arguments → LangChain.
- `st.session_state` remembers, `st.cache_resource` loads once, `st.write_stream` streams.

⬅️ [17 · Evaluation](17_evaluation.md) · 🏠 [00 · Big Picture](00_big_picture.md)
