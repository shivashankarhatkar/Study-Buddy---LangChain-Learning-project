# 📽️ Study Buddy: 5-Minute Demo Script

A step-by-step script for presenting the proof of concept (POC). The app has three pages, shown in the top menu: **💬 Ask**, **🎯 Quiz & Summary** and **⚙️ Settings**.

---

## ✅ Before the demo (10 minutes beforehand)

- [ ] `.env` has your Gemini key, `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true` and `LANGSMITH_PROJECT=study-buddy`
- [ ] Start the app: `streamlit run app/streamlit_app.py`
- [ ] Warm-up: ask one question, so the index is loaded and the first slow call is out of the way
- [ ] Have one **stakeholder-relevant PDF** ready (a syllabus, a chapter, a policy document)
- [ ] Optional: run `python step16_evaluation.py` once, so you can quote a quality score
- [ ] Keep the LangSmith project (`study-buddy`) open in a second browser tab
- [ ] Click **🧹 Clear chat** so the conversation starts empty

---

## 🎬 The walkthrough

| # | Do | Say |
|---|---|---|
| 1 | **⚙️ Settings** → **Add documents** → upload the PDF → **Add to library** | "Any PDF or notes file goes in. It's read, split into passages and indexed in seconds. Only the new file is processed." |
| 2 | Point to the **Answers** and **Quiz** settings | "How many passages the AI reads, how creative it is, and the quiz length, difficulty and source documents are all adjustable." |
| 3 | **💬 Ask** → ask a question about the PDF | "The answer streams in, and every fact carries a citation like **[1]**. It only uses the uploaded material." |
| 4 | Click **🔍 Under the hood** on that answer | "This shows the exact passages used, how well each one matched, how long each step took, and the token cost." |
| 5 | Ask a follow-up using "it" or "they" | "The earlier answer stays on screen. Study Buddy understands the follow-up from the conversation." Open Under the hood: "It rewrote the question before searching." |
| 6 | Ask something unrelated ("Who painted the Mona Lisa?") | "When the answer isn't in the material, it says so. No made-up answers. **The AI model wasn't even called.**" |
| 7 | **🎯 Quiz & Summary** → Quiz tab → enter a topic → **Generate quiz** | "Questions are generated from the documents, at the difficulty chosen in Settings. Some have more than one correct answer." |
| 8 | Tick answers → **✅ Evaluate** | "The quiz is scored instantly, showing the right answers and an explanation for each question." |
| 9 | **📝 Summary** tab → pick the PDF → **Summarize** | "A one-click revision sheet you can download. Earlier quizzes and summaries stay until you erase them." |
| 10 | Back on **💬 Ask**, Under the hood → **Open full trace in LangSmith ↗** | "Every answer is fully traceable for debugging and auditing. User 👍/👎 feedback lands right on the trace." |
| 11 | *(Optional)* Show the evaluation output | "A test set is graded automatically for correctness, made-up facts and honest refusals, so quality is measured, not assumed." |

**Closing line:** *"This is a working POC on a production-style architecture: citations, honest refusals, measured quality and full observability. Next steps are user accounts, a hosted database and deployment."*

---

## 🧯 If something goes wrong

| Problem | Quick fix |
|---|---|
| `429` / quota error | Add `GEMINI_REQUESTS_PER_MINUTE=10` to `.env`, restart the app, and keep the pace slower |
| Good questions get "not found" | **⚙️ Settings → Advanced** → lower **Minimum match score** (e.g. to 0.45) |
| Unrelated questions get answered | Raise **Minimum match score** a little (e.g. to 0.6) |
| A PDF gives no text | It's probably a scanned image; use a PDF with selectable text |
| No LangSmith button | Check that `LANGSMITH_API_KEY` is set and `LANGSMITH_TRACING=true` |

## 💬 Likely stakeholder questions

- **"Can it use our data?"** Yes. Any PDF or text file; nothing is trained, documents are only searched.
- **"How do we know it's accurate?"** Citations for every fact, plus an automated quality score that can be re-run after any change.
- **"What does it cost?"** Under the hood shows tokens per answer, and LangSmith adds them up across all usage.
- **"Can we switch AI providers?"** Yes. Model setup is isolated in one file (`config.py`).
