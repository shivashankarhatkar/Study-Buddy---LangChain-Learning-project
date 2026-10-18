"""
STEP 14  -  LangSmith Tracing: seeing what your app does, step by step
======================================================================
📖 Read first: learning_notes/14_langsmith_tracing.md

THE PROBLEM
    When Study Buddy gives a wrong answer, WHY did it happen?
        - Did the retriever find the wrong chunks?
        - Was the prompt badly filled in?
        - Did the model ignore the notes?
    With print() you only see the final answer. You can't see inside the chain.

THE SOLUTION: LangSmith
    LangSmith is a website (by the LangChain team) that records a "TRACE" of every
    run: each step, its input, its output, how long it took and how many tokens it used.

    A trace for "Ask a question about my notes" looks like a tree:

        Ask Notes (chain)                          2.1s
        ├── RunnableParallel
        │   ├── VectorStoreRetriever               0.3s  -> 4 documents
        │   └── RunnablePassthrough                       -> "What is Jupiter?"
        ├── context (format_docs)
        └── answer
            ├── ChatPromptTemplate                         -> 2 messages
            ├── ChatGoogleGenerativeAI             1.7s  -> AIMessage (312 tokens)
            └── StrOutputParser                           -> "Jupiter is..."

HOW TO TURN IT ON  (no code needed!)
    Put these in your .env file:
        LANGSMITH_TRACING=true
        LANGSMITH_API_KEY=your-key
        LANGSMITH_PROJECT=study-buddy
    Every LangChain Runnable (model, prompt, chain, retriever, tool, agent) is
    then traced AUTOMATICALLY. This file shows how to make traces easier to read.

Run this file:   python step14_langsmith_tracing.py
Then open https://smith.langchain.com -> Projects -> study-buddy
"""

import os

from langsmith import Client, trace, traceable, tracing_context

from config import print_tracing_status
from step05_chains_lcel import build_explain_chain, build_summary_chain
from step11_rag import build_rag_chain_with_sources, build_notes_quiz_chain


# ------------------------------------------------------------------
# 1. Automatic tracing: nothing special, just run a chain
# ------------------------------------------------------------------
def automatic_tracing_demo():
    # Because LANGSMITH_TRACING=true, this call is recorded automatically.
    # In LangSmith you'll see the full tree: retriever -> prompt -> model -> parser.
    result = build_rag_chain_with_sources().invoke("Why is Venus so hot?")
    print("Answer:", result["answer"])


# ------------------------------------------------------------------
# 2. Name, tag and label a single run with "config"
# ------------------------------------------------------------------
def config_demo():
    # Every .invoke() / .stream() / .batch() accepts a config dict:
    #   run_name -> the name shown in LangSmith (instead of "RunnableSequence")
    #   tags     -> labels you can FILTER by in LangSmith (e.g. show only "quiz" runs)
    #   metadata -> any extra key/value info you want to store with the run
    answer = build_explain_chain().invoke(
        {"topic": "gravity", "level": "beginner"},
        config={
            "run_name": "Explain Topic",
            "tags": ["study-buddy", "explain", "demo"],
            "metadata": {"student": "riya", "subject": "physics"},
        },
    )
    print(answer)


# ------------------------------------------------------------------
# 3. Give a chain a permanent name with .with_config()
# ------------------------------------------------------------------
# Instead of passing config on every call, you can "attach" it to the chain once.
# Every run of this chain will now appear as "Revision Summary" in LangSmith.
named_summary_chain = build_summary_chain().with_config(
    run_name="Revision Summary",
    tags=["study-buddy", "summary"],
)


def with_config_demo():
    print(named_summary_chain.invoke({"notes": "Water boils at 100 C at sea level. Ice melts at 0 C."}))


# ------------------------------------------------------------------
# 4. @traceable: trace YOUR OWN Python functions too
# ------------------------------------------------------------------
# LangChain objects are traced automatically, but normal Python functions are not.
# @traceable makes a function appear in LangSmith, and every LangChain call made
# INSIDE it is shown nested underneath it. Great for grouping one "user action":
#
#     Revision Session (your function)
#     ├── Ask Notes        (RAG chain)
#     └── Notes Quiz       (quiz chain)
#
@traceable(name="Revision Session", tags=["study-buddy", "session"])
def revision_session(topic: str) -> dict:
    """Answer a question about a topic, then make a quiz about it."""
    answer = build_rag_chain_with_sources().invoke(
        f"Explain {topic} in simple words.",
        config={"run_name": "Ask Notes"},
    )
    quiz = build_notes_quiz_chain().invoke(
        {"topic": topic, "num_questions": 2},
        config={"run_name": "Notes Quiz"},
    )
    # Whatever the function returns is shown as its "output" in LangSmith.
    return {"explanation": answer["answer"], "quiz_questions": [q.question for q in quiz.questions]}


# ------------------------------------------------------------------
# 5. Turn tracing off for one block of code
# ------------------------------------------------------------------
def no_tracing_demo():
    # Useful for private data or for quick tests you don't want to keep.
    with tracing_context(enabled=False):
        print(build_explain_chain().invoke({"topic": "magnets", "level": "beginner"}))
    print("(This run was NOT sent to LangSmith.)")


# ------------------------------------------------------------------
# 6. Wrapping several steps in ONE trace, and collecting user feedback
# ------------------------------------------------------------------
# The web app does "rewrite -> retrieve -> stream answer" as separate calls.
# "with trace(...) as run:" groups them under one parent run. Every LangChain
# call made inside the block is nested underneath it automatically.
# The run has an ID, which lets us attach user FEEDBACK (thumbs up/down) to it.
#
#     Ask Notes                 ← run (has run.id)    👍 user_rating = 1
#     ├── Rewrite Follow-up
#     ├── Retrieve Passages
#     └── Generate Answer
#
def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


def get_trace_url(run) -> str | None:
    """Link to this run in the LangSmith website (None if tracing is off or offline)."""
    if not tracing_enabled():
        return None
    try:
        return run.get_url()
    except Exception:
        return None


def send_feedback(run_id, thumbs_up: bool, comment: str | None = None):
    """Attach a user's 👍 / 👎 to a run. You can then filter runs by rating in LangSmith."""
    if tracing_enabled():
        Client().create_feedback(run_id=run_id, key="user_rating",
                                 score=1 if thumbs_up else 0, comment=comment)


def feedback_demo():
    with trace("Explain With Feedback", inputs={"topic": "friction"}, tags=["study-buddy", "demo"]) as run:
        answer = build_explain_chain().invoke({"topic": "friction", "level": "beginner"})
        run.end(outputs={"answer": answer})  # record what the whole block produced
    print(answer)
    send_feedback(run.id, thumbs_up=True, comment="Clear explanation")
    print("Trace URL:", get_trace_url(run))


if __name__ == "__main__":
    print_tracing_status()

    print("\n=== 1. Automatic tracing ===")
    automatic_tracing_demo()

    print("\n=== 2. run_name, tags and metadata ===")
    config_demo()

    print("\n=== 3. .with_config() ===")
    with_config_demo()

    print("\n=== 4. @traceable on our own function ===")
    print(revision_session("photosynthesis"))

    print("\n=== 5. Turning tracing off ===")
    no_tracing_demo()

    print("\n=== 6. One trace for several steps + feedback ===")
    feedback_demo()

    print("\nNow open https://smith.langchain.com and look at your 'study-buddy' project!")
