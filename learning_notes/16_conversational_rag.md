# 16 · Conversational RAG: Follow-ups, Scores, Filters & Citations

> **In one line:** the step 11 RAG chain works for single questions. This upgrade makes it work like a real product: it understands follow-ups, refuses honestly, can search selected files, and cites every fact.

📄 Code: [step15_conversational_rag.py](../step15_conversational_rag.py) · 🖥️ Used by the web app's **💬 Ask** page

---

## 🗺️ The upgraded flow

```
 question + chat history
        │
        ▼
 ① REWRITE ──────── "How many moons does it have?"  →  "How many moons does Jupiter have?"
        │
        ▼
 ② RETRIEVE ─────── with a FILTER (only the selected files) and optional MMR (varied results)
        │
        ▼
 ③ SCORE CHECK ──── drop passages below the minimum match score
        │
        ├── nothing left? ──► "I couldn't find that in your notes."   (no LLM call at all!)
        ▼
 ④ ANSWER ───────── numbered passages [1] [2] [3] → Gemini → "Jupiter has 95 moons [1]."
```

---

## ① Follow-up questions: rewriting the question

People don't repeat themselves in a conversation:

```
 You: "Tell me about Jupiter."
 You: "How many moons does it have?"     ← "it" = ???
```

Searching your notes for *"How many moons does **it** have?"* finds nothing useful, because the vector store doesn't know what "it" means. So **before** searching, we ask Gemini to rewrite the question using the chat history:

```
 ┌────────────── REWRITE_PROMPT ────────────────┐
 │ system: Rewrite the latest question so it    │
 │         makes sense WITHOUT the history.     │
 │         Do NOT answer it.                    │
 │ [history]  Human: Tell me about Jupiter.     │
 │            AI:    Jupiter is the largest...  │
 │ human:  How many moons does it have?         │
 └──────────────────────┬───────────────────────┘
                        ▼
        "How many moons does Jupiter have?"   ← this is what we search with
```

> 💡 It's skipped for the first question (no history), which saves an LLM call.

## ② Filters & MMR: controlling *what* is retrieved

**Filter:** a small function that says *yes* or *no* to each chunk, using its metadata:

```python
only_selected = lambda doc: doc.metadata["source"] in ["photosynthesis.txt"]
```

In the app, the **Documents to build quizzes from** setting builds this filter for quizzes.

**MMR (Maximal Marginal Relevance):** plain search can return 4 near-copies of the same paragraph. MMR picks passages that are relevant **and different from each other**:

```
 Plain similarity (k=3)              MMR (k=3)
 [1] Jupiter is the largest...       [1] Jupiter is the largest...
 [2] Jupiter, the largest planet...  [2] Its Great Red Spot is a storm...
 [3] The largest planet, Jupiter...  [3] Ganymede, its largest moon...
     ▲ repeats the same fact             ▲ more facts for the LLM
```

## ③ The minimum score: honest "not found" answers

Every passage has a **match score** (cosine similarity, step 09). Weak matches are thrown away:

```
 "Who painted the Mona Lisa?"
   [solar_system.md]   0.41  ✗ below 0.55
   [python_basics.md]  0.38  ✗ below 0.55
   → nothing left → "I couldn't find that in your notes."   ✅ no guessing
```

> 🎛️ Scores depend on the embedding model. Tune the threshold by looking at real scores in the app's **Under the hood** panel.

## ④ Numbered citations

Passages are numbered, and the prompt requires a citation after each fact:

```
 PASSAGES:                                         ANSWER:
 [1] (solar_system.md) Jupiter is the largest...   Jupiter is the largest planet [1]. Its Great Red
 [2] (solar_system.md) Its Great Red Spot is...    Spot is a storm bigger than Earth [2].
```

Anyone can check each sentence against its source. That's how a retrieval system earns trust.

---

## 🧩 Building blocks used

| Need | LangChain piece |
|---|---|
| Rewrite the follow-up | `ChatPromptTemplate` + `MessagesPlaceholder("history")` + `StrOutputParser` |
| Search with scores and a filter | `vector_store.similarity_search_with_score_by_vector(vector, k, filter=...)` |
| Varied results | `vector_store.max_marginal_relevance_search_by_vector(...)` |
| Streaming with token counts | `ANSWER_PROMPT \| llm` (no parser), adding the `AIMessageChunk`s together |
| See it all in LangSmith | `@traceable` + `with trace("Ask Notes") as run:` |

## ✅ Remember

- Rewrite follow-ups **before** searching.
- Filter by metadata, use MMR for variety, and drop weak matches with a minimum score.
- Number the passages and require citations.

⬅️ [15 · The Full App](15_full_app.md) · ➡️ **Next:** [17 · Evaluation](17_evaluation.md)
