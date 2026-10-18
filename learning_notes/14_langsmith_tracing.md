# 14 · LangSmith Tracing: An X-Ray for Your App

> **In one line:** LangSmith **records every step** your app takes (inputs, outputs, time, tokens) and shows it on a website, so you can **see inside** your chains and agents.

📄 Code: [step14_langsmith_tracing.py](../step14_langsmith_tracing.py) · 🌐 https://smith.langchain.com

---

## 🤔 The problem

Study Buddy gives a **wrong answer**. Why?

```
 You: "How tall is Olympus Mons?"
 Bot: "I couldn't find that in your notes."   🤨 ...but it IS in solar_system.md!

 Was it...
   ❓ the retriever finding the wrong chunks?
   ❓ the prompt being filled in badly?
   ❓ Gemini ignoring the notes?
```

With `print()` you only see the **final** answer. Everything in between is a black box.

## 💡 The idea: a flight recorder ✈️📼

Planes carry a "black box" that records everything. LangSmith is the black box for your LangChain app, except you can open it whenever you like.

```
        Your app (Study Buddy)                          LangSmith website
 ┌───────────────────────────────┐              ┌──────────────────────────────┐
 │ retriever ─► prompt ─► gemini │ ──traces──►  │ 📁 Project: study-buddy      │
 │      ─► parser                │  (sent in    │   ├─ Ask Notes      2.1s ✅  │
 └───────────────────────────────┘  background) │   ├─ Notes Quiz     3.4s ✅  │
                                                │   └─ Study Buddy Agent 5.0s ❌│
                                                └──────────────────────────────┘
```

---

## 🔌 Turning it on: zero code!

Add three lines to your `.env` file:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key        ← free at smith.langchain.com → Settings → API Keys
LANGSMITH_PROJECT=study-buddy               ← the folder your traces go into
```

That's it. `config.py` loads `.env`, and **every** model, prompt, chain, retriever, tool and agent is traced automatically.

---

## 🌳 What a trace looks like

Here's the trace for **app option 1 ("Ask Notes")**. Each row is one block from the chain, and you can click any row to see its exact input and output:

```
 ▼ Ask Notes                                   2.1s   tags: study-buddy, rag
   ▼ RunnableParallel<docs,question>           0.3s
       VectorStoreRetriever                    0.3s   in: "How tall is Olympus Mons?"
                                                      out: 4 Documents  ← 🔍 were they about Mars?
       RunnablePassthrough                            out: "How tall is Olympus Mons?"
     RunnableAssign<context>                          out: "[Source: solar_system.md] Mars is..."
   ▼ RunnableAssign<answer>                    1.8s
       ChatPromptTemplate                             out: [system: ...NOTES..., human: ...]
       ChatGoogleGenerativeAI                  1.7s   tokens: 812 in / 64 out
       StrOutputParser                                out: "Olympus Mons is about 22 km high..."
```

To debug the problem above, open the trace, click the **retriever** row, and look at which chunks it returned. Now you **know** where things went wrong.

And here's an **agent** trace (option 4), which shows every decision in the loop:

```
 ▼ Study Buddy Agent                             5.0s   tags: study-buddy, agent
     model    → tool_call: search_my_notes("largest planet")
     tools    → search_my_notes  ▸ VectorStoreRetriever
     model    → tool_call: save_flashcard(...)
     tools    → save_flashcard → "Flashcard saved."
     model    → "Jupiter is the largest planet..."
```

---

## 🏷️ Making traces easy to read

By default, a chain shows up with a technical name like `RunnableSequence`. Study Buddy gives each feature a **clear name** and **tags**:

| How | Code | Effect |
|---|---|---|
| per call | `.invoke(x, config={"run_name": "Ask Notes", "tags": ["rag"]})` | name and labels for this one run |
| metadata | `config={"metadata": {"student": "riya"}}` | extra info stored with the run |
| permanently | `chain.with_config(run_name="Revision Summary")` | every run of this chain gets the name |
| your own function | `@traceable(name="Revision Session")` | a normal Python function appears in the trace |
| turn off | `with tracing_context(enabled=False): ...` | this block is not recorded |

📍 In `main.py`, the helper `trace_config()` names every feature:

```
 Option 1 ─► "Ask Notes"          tags: [study-buddy, rag]
 Option 2 ─► "Summarize Notes"    tags: [study-buddy, summary]
 Option 3 ─► "Notes Quiz"         tags: [study-buddy, quiz]
 Option 4 ─► "Study Buddy Agent"  tags: [study-buddy, agent]
```

In LangSmith you can then **filter by tag**, for example "show me only the quiz runs".

### `@traceable`: grouping several chains into one trace

```python
@traceable(name="Revision Session")
def revision_session(topic):
    rag_chain.invoke(...)    # appears nested inside
    quiz_chain.invoke(...)   # appears nested inside
```

```
 ▼ Revision Session          ← your function
   ▶ Ask Notes               ← chain 1
   ▶ Notes Quiz              ← chain 2
```

---

### `with trace(...) as run`: one trace for several steps, plus user feedback

The web app runs *rewrite → retrieve → stream answer* as separate calls. Wrapping them in `trace()` groups them into **one** trace. The `run.id` then lets us attach the user's 👍/👎 to that exact answer:

```python
with trace("Ask Notes", inputs={"question": q}) as run:
    ...                                   # every LangChain call in here is nested under "Ask Notes"
    run.end(outputs={"answer": answer})

send_feedback(run.id, thumbs_up=True)     # → appears as "user_rating = 1" on the trace
```

```
 ▼ Ask Notes                 👍 user_rating: 1
   ├─ Rewrite Follow-up
   ├─ Retrieve Passages
   └─ Generate Answer
```

In LangSmith you can then filter for 👎 answers, and see exactly what went wrong.

---

## 📊 What else LangSmith shows you

- ⏱️ **Latency:** which step is slow?
- 🔢 **Tokens:** how much each call "costs"
- ❌ **Errors:** the exact step that failed, with the error message
- 💬 **Exact prompts:** the real text sent to Gemini, after the blanks were filled

## ⚠️ Good to know

| Point | Detail |
|---|---|
| Privacy | traces contain your prompts and notes; use `tracing_context(enabled=False)` for private data |
| Missing key | if `LANGSMITH_TRACING=true` but there's no key, `config.py` prints a warning |
| Don't want it? | set `LANGSMITH_TRACING=false`, and the app works exactly the same |

## ✅ Remember

- LangSmith = a recorder for every step of your LangChain app.
- Turn it on with 3 environment variables; no code needed.
- Use `run_name`, `tags`, `metadata`, `with_config` and `@traceable` to make traces readable.

⬅️ [13 · Agents](13_agents.md) · ➡️ **Next:** [15 · The Full App](15_full_app.md)
