# 15 · The Full App: How Study Buddy Uses Everything

> **In one line:** `main.py` adds **nothing new**. It connects the blocks from steps 01–14 into a menu.

📄 Code: [main.py](../main.py) · Run: `python main.py`

---

## 🖥️ The menu

```
==================================================
   STUDY BUDDY  -  your LangChain study assistant
==================================================
[LangSmith] Tracing is ON  -> see your traces at https://smith.langchain.com

What would you like to do?
  1. Ask a question about my notes
  2. Summarize a notes file
  3. Quiz me on a topic
  4. Chat with Study Buddy (agent)
  5. Rebuild the notes index (after adding new notes)
  6. Exit
```

---

## 1️⃣ Ask a question about my notes → **RAG**

```
 question ─► retriever ─► format_docs ─► RAG_PROMPT ─► Gemini ─► StrOutputParser ─► answer + sources
             (10, 11)       (11)           (03)         (01)        (04)
```

## 2️⃣ Summarize a notes file → **Loader + Chain + Streaming**

```
 notes/ ─► StudyNotesLoader ─► pick a file ─► SUMMARY_PROMPT | Gemini | StrOutputParser ─► .stream()
               (07)                                     (03, 05)                             (01)
```

## 3️⃣ Quiz me → **RAG + Structured output**

```
 topic ─► retriever (k=6) ─► material ─► QUIZ_PROMPT | Gemini.with_structured_output(Quiz) ─► Quiz
            (11)                             (05)              (04)
                                                                                              │
                                     main.py loops over quiz.questions, checks your answers ◄─┘
```

## 4️⃣ Chat → **Agent + Tools + Memory**

```
 your message ─► create_agent ──loop──► search_my_notes / calculate / save_flashcard / show_flashcards
                    (13)                                   (12)
                    └── InMemorySaver remembers the chat (thread_id)  (06, 13)
```

## 5️⃣ Rebuild the index → **Load, Split, Embed, Store**

```
 notes/ ─► load (07) ─► split (08) ─► embed (09) ─► InMemoryVectorStore (10) ─► 💾 data/notes_index.json
```

## 🔭 All of it → **LangSmith** (14)

Each option passes `config=trace_config("Ask Notes", "rag")` and so on, so every run shows up in LangSmith with a clear name and tags.

---

## 🧭 Which concept powers which feature?

| Concept ↓ / Feature → | 1 Ask | 2 Summary | 3 Quiz | 4 Chat | 5 Rebuild |
|---|:-:|:-:|:-:|:-:|:-:|
| 01 Chat model | ✅ | ✅ | ✅ | ✅ | |
| 02 Messages | ✅ | ✅ | ✅ | ✅ | |
| 03 Prompt templates | ✅ | ✅ | ✅ | | |
| 04 Parsers / structured | ✅ | ✅ | ✅ | | |
| 05 LCEL chains | ✅ | ✅ | ✅ | | |
| 06 Memory | | | | ✅ | |
| 07 Loaders | | ✅ | | | ✅ |
| 08 Splitters | | | | | ✅ |
| 09 Embeddings | ✅ | | ✅ | ✅ | ✅ |
| 10 Vector store | ✅ | | ✅ | ✅ | ✅ |
| 11 Retriever / RAG | ✅ | | ✅ | ✅ | |
| 12 Tools | | | | ✅ | |
| 13 Agent | | | | ✅ | |
| 14 LangSmith | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 🚀 Ideas to practise on your own

Each idea uses what you've learned. Try them one at a time:

1. **Easy:** change the system prompt so Study Buddy answers like a pirate 🏴‍☠️ (step 03).
2. **Easy:** add your own class notes (PDF) to `notes/` and rebuild the index (steps 07–10).
3. **Medium:** add a `difficulty` field ("easy", "hard") to the `Quiz` model (step 04).
4. **Medium:** add a new tool, `get_today_date()`, so the agent can count the days to your exam (step 12).
5. **Medium:** make option 1 remember earlier questions using `RunnableWithMessageHistory` (step 06).
6. **Challenge:** add a menu option "Explain like I'm 5" using a new prompt template and chain, and give it its own `run_name` in LangSmith.

🎉 **Congratulations, you now know the core of LangChain!**

⬅️ [14 · LangSmith Tracing](14_langsmith_tracing.md) · ➡️ **Next (production upgrades):** [16 · Conversational RAG](16_conversational_rag.md)
