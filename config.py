"""
config.py  -  Shared settings for the whole project
=====================================================

Every other file imports its LLM and embedding model from HERE.
That way, if you want to switch models, you only change ONE file.

LangChain idea to notice:
    LangChain gives every chat model the SAME interface (.invoke, .stream, .batch).
    So if one day you switch from Gemini to OpenAI or Claude, you would only change
    the two functions below - the rest of the project keeps working untouched.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Gemini's chat model class and embedding model class.
# They live in the "langchain-google-genai" integration package.
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.rate_limiters import InMemoryRateLimiter


# ------------------------------------------------------------------
# 1. Load the API key from the .env file
# ------------------------------------------------------------------
# load_dotenv() reads the .env file and puts GOOGLE_API_KEY into os.environ.
# The Gemini classes automatically look for GOOGLE_API_KEY, so we never
# have to pass the key around by hand.
load_dotenv()

# The Gemini classes accept the key under either name: GOOGLE_API_KEY or GEMINI_API_KEY.
GEMINI_KEY_FOUND = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))

if not GEMINI_KEY_FOUND:
    print("WARNING: No Gemini API key found. Copy .env.example to .env and add GOOGLE_API_KEY.")


# ------------------------------------------------------------------
# 1b. LangSmith tracing (see step14_langsmith_tracing.py)
# ------------------------------------------------------------------
# There is NO tracing code to write! LangChain checks these environment variables
# (loaded from .env above) and, if LANGSMITH_TRACING is "true", automatically sends
# a record ("trace") of every model call, chain, retriever and tool to LangSmith:
#     LANGSMITH_TRACING=true
#     LANGSMITH_API_KEY=...
#     LANGSMITH_PROJECT=study-buddy     (the folder your traces are grouped in)
if os.getenv("LANGSMITH_TRACING", "").lower() == "true" and not os.getenv("LANGSMITH_API_KEY"):
    print("WARNING: LANGSMITH_TRACING is on but LANGSMITH_API_KEY is missing. "
          "Add the key to .env or set LANGSMITH_TRACING=false.")


def print_tracing_status():
    """Tell the user whether LangSmith tracing is on, and where to look."""
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        project = os.getenv("LANGSMITH_PROJECT", "default")
        print(f"[LangSmith] Tracing is ON  -> see your traces at https://smith.langchain.com (project: '{project}')")
    else:
        print("[LangSmith] Tracing is OFF (set LANGSMITH_TRACING=true in .env to turn it on)")


# ------------------------------------------------------------------
# 2. Project settings (change these freely)
# ------------------------------------------------------------------
CHAT_MODEL_NAME = "gemini-3.5-flash"         # the LLM that writes answers
EMBEDDING_MODEL_NAME = "gemini-embedding-001"  # the model that turns text into numbers (vectors)

# All paths are built from the folder this file lives in, so the project works
# no matter which folder you start it from (e.g. the Streamlit app in app/).
PROJECT_DIR = Path(__file__).parent
NOTES_FOLDER = str(PROJECT_DIR / "notes")                       # your study notes (.txt / .md / .pdf)
VECTOR_STORE_PATH = str(PROJECT_DIR / "data" / "notes_index.json")  # the saved searchable index
FLASHCARDS_FILE = str(PROJECT_DIR / "data" / "flashcards.txt")      # flashcards saved by the app/agent
EVAL_DATASET_PATH = str(PROJECT_DIR / "evaluation" / "test_questions.json")  # quality test set (step 16)
EVAL_RESULTS_PATH = str(PROJECT_DIR / "data" / "eval_results.json")          # latest quality score

# Optional: limit how many LLM requests per minute we send.
# The free Gemini tier only allows a few requests per minute. If you see
# "429 / quota exceeded" errors, add e.g. GEMINI_REQUESTS_PER_MINUTE=10 to .env.
# InMemoryRateLimiter is a LangChain helper that makes calls WAIT their turn.
_rpm = os.getenv("GEMINI_REQUESTS_PER_MINUTE")
RATE_LIMITER = InMemoryRateLimiter(requests_per_second=float(_rpm) / 60) if _rpm else None


# ------------------------------------------------------------------
# 3. Helper functions that create the models
# ------------------------------------------------------------------
def get_chat_model(temperature: float = 0.3):
    """
    Create and return the chat model (the "LLM").

    temperature controls creativity:
        0.0  -> very focused, gives almost the same answer every time
        1.0  -> more creative / random
    For a study assistant we want mostly factual answers, so we keep it low.
    """
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL_NAME,
        temperature=temperature,
        rate_limiter=RATE_LIMITER,  # None = no limit
    )


def get_embeddings():
    """
    Create and return the embedding model.

    An embedding model does NOT chat. It converts a piece of text into a list
    of numbers (a "vector") that represents its MEANING. Texts with similar
    meanings get similar vectors. We use this later to search our notes.
    """
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
