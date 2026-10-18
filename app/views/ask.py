"""
Ask  -  chat with your notes (conversational RAG, see step15_conversational_rag.py)

- Every question and answer stays on screen; new ones are added below.
- Follow-up questions work because the whole chat is passed as history.
- "Under the hood" (passages, scores, timings, trace) is hidden until you click it.
- Settings (passages to retrieve, temperature, ...) come from the Settings page.
"""

import time

import streamlit as st
from langsmith import trace

from shared import get_vector_store, page_header
from step14_langsmith_tracing import tracing_enabled, get_trace_url, send_feedback
from step15_conversational_rag import prepare_answer, build_answer_chain, to_history, NOT_FOUND_MESSAGE

EXAMPLE_QUESTIONS = [
    "Why is Venus hotter than Mercury?",
    "Tell me about Jupiter",
    "What are the products of photosynthesis?",
    "Who painted the Mona Lisa?",  # not in the notes -> shows an honest "not found"
]

if "chat" not in st.session_state:
    st.session_state.chat = []            # [{"role", "content", "details"}] = the conversation memory
    st.session_state.feedback_sent = set()
chat = st.session_state.chat
settings = st.session_state


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
title_col, button_col = st.columns([5, 1], vertical_alignment="bottom")
with title_col:
    page_header("💬 Ask your notes", "Answers come only from your documents, with numbered citations [1] [2].")
with button_col:
    if chat and st.button("🧹 Clear chat", width="stretch"):
        st.session_state.chat = []
        st.rerun()


# ------------------------------------------------------------------
# "Under the hood" (hidden inside an expander until clicked)
# ------------------------------------------------------------------
def location(p: dict) -> str:
    return p["source"] + (f" · page {p['page']}" if p.get("page") else "")


def pipeline_diagram(details: dict) -> str:
    """A Graphviz (DOT) diagram of the RAG steps for one answer."""
    grey = 'fillcolor="#F3F4F6", fontcolor="#9CA3AF"'
    rewrite_style = "" if details["rewritten"] else grey
    llm_style = "" if details["passages"] else grey
    return f"""
    digraph {{
        rankdir=LR; bgcolor="transparent";
        node [shape=box, style="rounded,filled", color="#4F46E5", fontname="Helvetica", fontsize=10, fillcolor="#EEF2FF"];
        edge [color="#6B7280", arrowsize=0.6];
        q [label="Question"];
        r [label="Rewrite\\nfollow-up", {rewrite_style}];
        s [label="Retrieve\\n{len(details['passages'])} passages"];
        g [label="Gemini\\n{details['generation_s']:.1f}s", {llm_style}];
        a [label="Answer\\n+ citations"];
        q -> r -> s -> g -> a;
    }}"""


def show_under_the_hood(details: dict):
    with st.expander("🔍 Under the hood"):
        st.graphviz_chart(pipeline_diagram(details), width="stretch")

        if details["rewritten"]:
            st.info(f"✨ **Follow-up rewritten** before searching:\n\n“{details['standalone_question']}”")

        m1, m2, m3 = st.columns(3)
        m1.metric("Retrieval", f"{details['retrieval_s']:.1f}s")
        m2.metric("Generation", f"{details['generation_s']:.1f}s")
        m3.metric("Tokens", f"{details['tokens']:,}" if details.get("tokens") else "–")

        if not details["passages"]:
            st.warning("No passage passed the minimum match score, so the answer is “not found” "
                       "and **the LLM was not called**. Nothing was made up.")
        for p in details["passages"]:
            st.progress(min(max(p["score"], 0.0), 1.0), text=f"**[{p['number']}] {location(p)}** · match {p['score']:.2f}")
            st.caption(p["text"] if len(p["text"]) < 400 else p["text"][:400] + "…")

        if details.get("trace_url"):
            st.link_button("Open full trace in LangSmith ↗", details["trace_url"])


def show_feedback(index: int, details: dict):
    """👍/👎 buttons. The rating is attached to this answer's LangSmith trace."""
    if not (tracing_enabled() and details.get("run_id")):
        return
    rating = st.feedback("thumbs", key=f"feedback_{index}")
    if rating is not None and index not in st.session_state.feedback_sent:
        send_feedback(details["run_id"], thumbs_up=(rating == 1))
        st.session_state.feedback_sent.add(index)
        st.toast("Thanks! Feedback saved to LangSmith.", icon="🙏")


# ------------------------------------------------------------------
# Answering a new question: rewrite -> retrieve -> stream the answer
# ------------------------------------------------------------------
def answer(question: str, history: list) -> dict:
    # Everything inside this "with" block becomes ONE trace in LangSmith (see step 14).
    with trace("Ask Notes", inputs={"question": question}, tags=["study-buddy", "rag", "web-app"]) as run:
        start = time.perf_counter()
        with st.spinner("Searching your notes…"):
            prepared = prepare_answer(get_vector_store(), question, history, k=settings.k,
                                      diverse=settings.diverse, min_score=settings.min_score)
        retrieved_at = time.perf_counter()

        final_message = None
        if prepared["passages"]:
            def stream_answer():
                # .stream() gives AIMessageChunks. We show each piece of text immediately and
                # add the chunks together, so at the end we also know the token usage.
                nonlocal final_message
                chain = build_answer_chain(temperature=settings.temperature)
                for chunk in chain.stream(prepared["chain_input"], config={"run_name": "Generate Answer"}):
                    final_message = chunk if final_message is None else final_message + chunk
                    yield chunk.text
            answer_text = st.write_stream(stream_answer())
        else:
            answer_text = NOT_FOUND_MESSAGE
            st.markdown(answer_text)

        run.end(outputs={"answer": answer_text})

    usage = final_message.usage_metadata if final_message is not None else None
    return {
        "answer": answer_text,
        "standalone_question": prepared["standalone_question"],
        "rewritten": prepared["standalone_question"].strip(" ?.!").lower() != question.strip(" ?.!").lower(),
        "passages": prepared["passages"],
        "retrieval_s": retrieved_at - start,
        "generation_s": time.perf_counter() - retrieved_at,
        "tokens": usage["total_tokens"] if usage else None,
        "run_id": str(run.id),
        "trace_url": get_trace_url(run),
    }


# ------------------------------------------------------------------
# The conversation (oldest at the top, newest at the bottom)
# ------------------------------------------------------------------
if not chat:
    st.markdown("👋 **Ask anything about your notes.** Try one of these:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        if cols[i].button(example, key=f"example_{i}", width="stretch"):
            st.session_state.pending_question = example
            st.rerun()
    st.caption("Tip: after asking about Jupiter, ask “How many moons does it have?” to try a follow-up.")

for i, message in enumerate(chat):
    with st.chat_message(message["role"], avatar="🧑‍🎓" if message["role"] == "user" else "📚"):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_under_the_hood(message["details"])
            show_feedback(i, message["details"])

# st.chat_input on the main page is always pinned to the bottom of the screen.
question = st.chat_input("Ask a question about your notes…") or st.session_state.pop("pending_question", None)
if question:
    history = to_history(chat)  # the conversation so far -> lets follow-up questions work
    chat.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="📚"):
        details = answer(question, history)
    chat.append({"role": "assistant", "content": details.pop("answer"), "details": details})
    st.rerun()  # redraw the whole conversation, including this answer's "Under the hood" button
