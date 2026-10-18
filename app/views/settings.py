"""
Settings  -  documents, answer settings and quiz settings

Every widget here has a "key". Its value is stored in st.session_state under that key,
which is where the Ask and Quiz & Summary pages read it from.
"""

import os

import streamlit as st

from config import NOTES_FOLDER, CHAT_MODEL_NAME, EMBEDDING_MODEL_NAME
from shared import get_vector_store, page_header, reset_settings
from step10_vector_store import add_file_to_store, remove_source_from_store, list_sources
from step14_langsmith_tracing import tracing_enabled

vector_store = get_vector_store()
library = list_sources(vector_store)   # {file name: number of chunks}

page_header("⚙️ Settings", "Changes apply immediately to the Ask and Quiz & Summary pages.")


# ------------------------------------------------------------------
# Documents
# ------------------------------------------------------------------
st.markdown("### 📁 Documents")
docs_col, upload_col = st.columns(2, gap="large")

with upload_col:
    files = st.file_uploader("Add documents (PDF, TXT or Markdown)", type=["pdf", "txt", "md"],
                             accept_multiple_files=True)
    if files and st.button("Add to library", type="primary"):
        for uploaded in files:
            with st.spinner(f"Reading and indexing {uploaded.name}…"):
                # 1) save the file into notes/   2) load -> split -> embed -> store (step 10)
                path = os.path.join(NOTES_FOLDER, uploaded.name)
                with open(path, "wb") as out:
                    out.write(uploaded.getbuffer())
                remove_source_from_store(vector_store, uploaded.name)  # replace if it already existed
                chunks = add_file_to_store(vector_store, path)
            st.toast(f"Added {uploaded.name} ({chunks} chunks)", icon="✅")
        st.rerun()

with docs_col:
    st.caption(f"{len(library)} documents · {sum(library.values())} chunks in the library")
    if not library:
        st.info("No documents yet. Add some on the right.")
    for name, count in library.items():
        name_col, delete_col = st.columns([5, 1], vertical_alignment="center")
        name_col.markdown(f"📄 **{name}** · {count} chunks")
        if delete_col.button("🗑️", key=f"delete_{name}", help=f"Remove {name}"):
            remove_source_from_store(vector_store, name)
            file_path = os.path.join(NOTES_FOLDER, name)
            if os.path.exists(file_path):
                os.remove(file_path)
            st.rerun()

st.divider()


# ------------------------------------------------------------------
# Answers
# ------------------------------------------------------------------
st.markdown("### 💬 Answers")
c1, c2 = st.columns(2, gap="large")
c1.slider("Documents (passages) to retrieve per question", 1, 10, key="k",
          help="How many of the best-matching passages from your notes are given to the AI.")
c2.slider("Temperature", 0.0, 1.0, step=0.1, key="temperature",
          help="0 = focused and consistent answers · 1 = more creative. Also used for quizzes and summaries.")

with st.expander("Advanced"):
    a1, a2 = st.columns(2, gap="large")
    a1.slider("Minimum match score", 0.0, 0.9, step=0.05, key="min_score",
              help="Passages scoring below this are ignored. If none are left, Study Buddy says it "
                   "couldn't find the answer instead of guessing. Lower it if good questions get 'not found'.")
    a2.toggle("Diverse passages (MMR)", key="diverse",
              help="Pick passages that are relevant AND different from each other, instead of near-duplicates.")

st.divider()


# ------------------------------------------------------------------
# Quiz
# ------------------------------------------------------------------
st.markdown("### 🎯 Quiz")
q1, q2 = st.columns(2, gap="large")
q1.slider("Number of questions", 1, 10, key="quiz_questions")
q2.select_slider("Difficulty", options=["Easy", "Medium", "Hard"], key="quiz_difficulty")

# Forget documents that were deleted, so the box below only shows real files
st.session_state.quiz_sources = [s for s in st.session_state.quiz_sources if s in library]
st.multiselect("Documents to build quizzes from", options=list(library), key="quiz_sources",
               placeholder="All documents")

st.divider()
st.button("↺ Reset to defaults", on_click=reset_settings)

st.caption(f"Model: `{CHAT_MODEL_NAME}` · Embeddings: `{EMBEDDING_MODEL_NAME}` · "
           + ("🟢 LangSmith tracing on" if tracing_enabled() else "⚪ LangSmith tracing off"))
