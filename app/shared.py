"""
shared.py  -  Things every page of the web app needs
====================================================

The web app is only a FRONTEND. All the LangChain logic lives in the step files
(step10 = vector store, step15 = conversational RAG, ...). This file:
    - makes the step files importable from the app/ folder
    - loads the vector store ONCE and keeps it in memory (st.cache_resource)
    - holds the user's settings (chosen on the Settings page, used by the other pages)
"""

import sys
from pathlib import Path

# The step files live one folder up (the project root). Add it to Python's search path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from step10_vector_store import load_vector_store
from step15_conversational_rag import DEFAULT_MIN_SCORE


# ------------------------------------------------------------------
# The vector store, loaded once per server (not on every click)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading your notes library...")
def get_vector_store():
    # Streamlit re-runs the page script on every click. cache_resource makes sure
    # this (slow) load happens only once, and every page shares the same object.
    return load_vector_store()


# ------------------------------------------------------------------
# Settings (edited on the Settings page)
# ------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "k": 4,                           # passages retrieved per question
    "temperature": 0.3,               # creativity of the answers
    "min_score": DEFAULT_MIN_SCORE,   # passages below this match score are ignored
    "diverse": False,                 # MMR: varied passages instead of near-duplicates
    "quiz_questions": 5,
    "quiz_difficulty": "Medium",
    "quiz_sources": [],               # documents to build quizzes from ([] = all)
}


def init_settings():
    """
    Called on every run. Setting each value again stops Streamlit from forgetting
    a Settings-page widget's value while you are on another page.
    """
    for key, value in DEFAULT_SETTINGS.items():
        st.session_state[key] = st.session_state.get(key, value)


def reset_settings():
    for key, value in DEFAULT_SETTINGS.items():
        st.session_state[key] = value


def page_header(title: str, subtitle: str):
    st.markdown(f"## {title}")
    st.caption(subtitle)
