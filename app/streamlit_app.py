"""
streamlit_app.py  -  The Study Buddy web app (entry point)
==========================================================

Run from the project root:
    streamlit run app/streamlit_app.py

Streamlit re-runs this script from top to bottom on every click. It:
    1. sets up the page and the user's settings
    2. shows whichever of the three pages is selected in the top menu
"""

import streamlit as st

st.set_page_config(page_title="Study Buddy", page_icon="📚", layout="wide")

from shared import init_settings  # noqa: E402  (must come after set_page_config)
from config import GEMINI_KEY_FOUND  # noqa: E402

if not GEMINI_KEY_FOUND:
    st.error("**Gemini API key is missing.** Add `GOOGLE_API_KEY=...` (or `GEMINI_API_KEY=...`) to your "
             "`.env` file, then restart the app.")
    st.stop()

init_settings()

navigation = st.navigation(
    [
        st.Page("views/ask.py", title="Ask", icon="💬", default=True),
        st.Page("views/study.py", title="Quiz & Summary", icon="🎯"),
        st.Page("views/settings.py", title="Settings", icon="⚙️"),
    ],
    position="top",
)
navigation.run()
