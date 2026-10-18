"""
Quiz & Summary  -  two tabs, each keeping everything it generated until you press "Erase"

Quiz:    retrieve passages (step 15) -> QUIZ_PROMPT | Gemini.with_structured_output(Quiz)  (steps 04, 05)
         Questions may have one or more correct answers. Tick your answers, press Evaluate.
Summary: load_file (step 07) -> SUMMARY_PROMPT | Gemini | StrOutputParser, streamed  (step 05)

Quiz settings (number of questions, difficulty, documents) come from the Settings page.
"""

import os
import uuid

import streamlit as st

from config import NOTES_FOLDER
from shared import get_vector_store, page_header
from step04_output_parsers import answers_match
from step05_chains_lcel import build_quiz_chain, build_summary_chain
from step07_document_loaders import load_file
from step10_vector_store import list_sources
from step15_conversational_rag import retrieve_passages, format_numbered_passages

settings = st.session_state
quizzes = st.session_state.setdefault("quizzes", [])      # newest first; kept until "Erase"
summaries = st.session_state.setdefault("summaries", [])  # newest first; kept until "Erase"
library = list(list_sources(get_vector_store()))

page_header("🎯 Quiz & Summary", "Test yourself and revise, using questions and summaries built from your own documents.")

if not library:
    st.info("Your library is empty. Add documents on the ⚙️ Settings page first.")
    st.stop()

quiz_tab, summary_tab = st.tabs(["🎯 Quiz", "📝 Summary"])


# ------------------------------------------------------------------
# Quiz tab
# ------------------------------------------------------------------
def generate_quiz(topic: str):
    sources = [s for s in settings.quiz_sources if s in library] or None  # None = all documents
    with st.spinner("Reading your notes and writing questions…"):
        passages = retrieve_passages(get_vector_store(), topic, k=8, sources=sources, min_score=0)
        quiz = build_quiz_chain(temperature=settings.temperature).invoke(
            {"material": format_numbered_passages(passages), "topic": topic,
             "num_questions": settings.quiz_questions, "difficulty": settings.quiz_difficulty.lower()},
            config={"run_name": "Notes Quiz", "tags": ["study-buddy", "quiz", "web-app"]},
        )
    quizzes.insert(0, {
        "id": uuid.uuid4().hex[:8],
        "topic": topic,
        "difficulty": settings.quiz_difficulty,
        "sources": sorted({p["source"] for p in passages}),
        "questions": [q.model_dump() for q in quiz.questions],   # Pydantic objects -> plain dicts
        "answers": [[] for _ in quiz.questions],                  # the options the student ticked
        "evaluated": False,
    })


def show_quiz(quiz: dict):
    with st.container(border=True):
        st.markdown(f"#### {quiz['topic']}")
        st.caption(f"{quiz['difficulty']} · {len(quiz['questions'])} questions · from {', '.join(quiz['sources'])}")

        if quiz["evaluated"]:
            score = sum(answers_match(chosen, q["correct_answers"])
                        for chosen, q in zip(quiz["answers"], quiz["questions"]))
            total = len(quiz["questions"])
            st.metric("Score", f"{score} / {total}", f"{score / total:.0%}", delta_color="off")

        for i, q in enumerate(quiz["questions"]):
            st.markdown(f"**Q{i + 1}. {q['question']}**")
            if not quiz["evaluated"]:
                st.caption("Select all that apply")

            correct = {c.strip().lower() for c in q["correct_answers"]}
            chosen = []
            for j, option in enumerate(q["options"]):
                key = f"{quiz['id']}_{i}_{j}"
                if key not in st.session_state:  # restore ticks, e.g. after visiting another page
                    st.session_state[key] = option in quiz["answers"][i]

                label = option
                if quiz["evaluated"]:
                    if option.strip().lower() in correct:
                        label += "  ✅"
                    elif st.session_state[key]:
                        label += "  ❌"
                if st.checkbox(label, key=key, disabled=quiz["evaluated"]):
                    chosen.append(option)
            quiz["answers"][i] = chosen

            if quiz["evaluated"]:
                if answers_match(chosen, q["correct_answers"]):
                    st.success(f"Correct! {q['explanation']}")
                else:
                    st.error(f"Correct answer(s): **{', '.join(q['correct_answers'])}**. {q['explanation']}")

        if not quiz["evaluated"]:
            if st.button("✅ Evaluate", key=f"evaluate_{quiz['id']}", type="primary"):
                quiz["evaluated"] = True
                st.rerun()
        elif st.button("🔁 Retake", key=f"retake_{quiz['id']}"):
            quiz["evaluated"] = False
            quiz["answers"] = [[] for _ in quiz["questions"]]
            for i, q in enumerate(quiz["questions"]):
                for j in range(len(q["options"])):
                    st.session_state.pop(f"{quiz['id']}_{i}_{j}", None)
            st.rerun()


with quiz_tab:
    chosen_docs = [s for s in settings.quiz_sources if s in library]
    st.caption(f"⚙️ {settings.quiz_questions} questions · {settings.quiz_difficulty} · "
               f"from {', '.join(chosen_docs) or 'all documents'}  (change these in Settings)")

    with st.form("new_quiz"):
        c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
        topic = c1.text_input("Quiz topic", placeholder="e.g. the planets, photosynthesis, Python loops")
        generate = c2.form_submit_button("✨ Generate quiz", type="primary", width="stretch")
    if generate and topic.strip():
        generate_quiz(topic.strip())

    if quizzes:
        if st.button("🗑️ Erase all quizzes", key="erase_quizzes"):
            quizzes.clear()
            st.rerun()
        for quiz in quizzes:
            show_quiz(quiz)


# ------------------------------------------------------------------
# Summary tab
# ------------------------------------------------------------------
with summary_tab:
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    chosen_file = c1.selectbox("Document to summarize", library)
    summarize = c2.button("✨ Summarize", type="primary", width="stretch")

    if summarize:
        path = os.path.join(NOTES_FOLDER, chosen_file)
        if not os.path.exists(path):
            st.error(f"{chosen_file} is in the index but the file is missing from the notes folder.")
        else:
            notes_text = "\n".join(doc.page_content for doc in load_file(path))
            with st.container(border=True):
                st.markdown(f"#### 📄 {chosen_file}")
                text = st.write_stream(build_summary_chain(temperature=settings.temperature).stream(
                    {"notes": notes_text},
                    config={"run_name": "Summarize Notes", "tags": ["study-buddy", "summary", "web-app"]},
                ))
            summaries.insert(0, {"id": uuid.uuid4().hex[:8], "file": chosen_file, "text": text})
            st.rerun()

    if summaries:
        if st.button("🗑️ Erase all summaries", key="erase_summaries"):
            summaries.clear()
            st.rerun()
        for summary in summaries:
            with st.container(border=True):
                st.markdown(f"#### 📄 {summary['file']}")
                st.markdown(summary["text"])
                st.download_button("⬇️ Download", summary["text"], key=f"download_{summary['id']}",
                                   file_name=f"{os.path.splitext(summary['file'])[0]}_summary.md",
                                   mime="text/markdown")
