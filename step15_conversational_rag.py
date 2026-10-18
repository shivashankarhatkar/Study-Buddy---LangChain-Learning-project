"""
STEP 15  -  Conversational RAG: follow-ups, scores, filters and citations
=========================================================================
📖 Read first: learning_notes/16_conversational_rag.md  (simple explanation with diagrams)

The RAG chain from step 11 works, but a real product needs more. This file
upgrades it with five improvements. The web app's "Ask" page uses it.

  1. FOLLOW-UP QUESTIONS  (history-aware question rewriting)
       You: "Tell me about Jupiter."   ...   You: "How many moons does it have?"
       Searching the notes for "How many moons does IT have?" fails - what is "it"?
       So we first ask the LLM to REWRITE the follow-up into a standalone question:
       "How many moons does Jupiter have?"  ->  then we search with that.

  2. MATCH SCORES + A MINIMUM SCORE  (honest "not found" answers)
       Every retrieved passage gets a similarity score (step 09). Passages below a
       minimum score are thrown away. If nothing is left, we don't even call the
       LLM - we answer "I couldn't find that in your notes."

  3. FILTERS  (search only some documents)
       A filter function decides which chunks are allowed, using their metadata.

  4. DIVERSE RESULTS  (MMR = Maximal Marginal Relevance)
       Plain similarity search may return 4 near-identical chunks. MMR picks chunks
       that are relevant AND different from each other, so the LLM sees more facts.

  5. NUMBERED CITATIONS
       Passages are numbered [1], [2]... and the LLM must cite them, so every
       sentence in the answer can be checked against its source.

THE FLOW:
    question + history ─► rewrite ─► retrieve (filter, MMR, scores) ─► min-score check
                                                                         │ none left → "not found"
                                                                         ▼
                                        numbered passages ─► ANSWER_PROMPT ─► Gemini ─► answer [1][2]

Run this file:   python step15_conversational_rag.py
"""

from langsmith import traceable
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from config import get_chat_model
from step09_embeddings import cosine_similarity
from step10_vector_store import load_vector_store


NOT_FOUND_MESSAGE = "I couldn't find that in your notes. Try rephrasing, or upload notes that cover this topic."

# Passages scoring below this are ignored. Scores depend on the embedding model,
# so tune it by looking at real scores (the web app shows them in "Under the hood").
DEFAULT_MIN_SCORE = 0.55

# How many earlier messages we send along (more = better context, but more tokens)
HISTORY_WINDOW = 6


def hide_vector_store(inputs: dict) -> dict:
    """For LangSmith: don't upload the whole index (thousands of numbers) as a trace input."""
    return {key: value for key, value in inputs.items() if key != "vector_store"}


# ------------------------------------------------------------------
# 1. Rewriting follow-up questions
# ------------------------------------------------------------------
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You rewrite a student's latest question so it can be understood WITHOUT the chat history. "
     "Replace words like 'it', 'they' or 'that' with what they refer to. "
     "Do NOT answer the question. If it is already standalone, return it unchanged. "
     "Return ONLY the rewritten question."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


def rewrite_question(question: str, history: list) -> str:
    """Turn a follow-up into a standalone question (skipped when there is no history)."""
    if not history:
        return question  # first question: nothing to rewrite, so we save an LLM call

    rewriter = REWRITE_PROMPT | get_chat_model(temperature=0) | StrOutputParser()
    return rewriter.invoke(
        {"history": history[-HISTORY_WINDOW:], "question": question},
        config={"run_name": "Rewrite Follow-up"},
    ).strip()


# ------------------------------------------------------------------
# 2-4. Retrieval with scores, filters and MMR
# ------------------------------------------------------------------
@traceable(name="Retrieve Passages", process_inputs=hide_vector_store)  # one step in LangSmith
def retrieve_passages(vector_store, query: str, k: int = 4, sources: list[str] | None = None,
                      diverse: bool = False, min_score: float = DEFAULT_MIN_SCORE) -> list[dict]:
    """
    Search the notes and return a list of passages like:
        {"number": 1, "source": "solar_system.md", "page": None, "score": 0.82, "text": "..."}
    """
    # Embed the question ONCE and reuse the vector for searching and scoring.
    query_vector = vector_store.embedding.embed_query(query)

    # A filter is just a function: Document -> True (keep) / False (skip).
    only_selected = (lambda doc: doc.metadata.get("source") in sources) if sources else None

    if diverse:
        # MMR: fetch 4x more candidates, then pick k that are relevant AND varied.
        docs = vector_store.max_marginal_relevance_search_by_vector(
            query_vector, k=k, fetch_k=k * 4, filter=only_selected)
        # MMR doesn't return scores, so we compute them with our step 09 function.
        scored = [(doc, cosine_similarity(query_vector, vector_store.store[doc.id]["vector"])) for doc in docs]
    else:
        scored = vector_store.similarity_search_with_score_by_vector(query_vector, k=k, filter=only_selected)

    passages = []
    for doc, score in scored:
        if score >= min_score:  # drop weak matches
            passages.append({
                "number": len(passages) + 1,
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "score": round(float(score), 3),
                "text": doc.page_content,
            })
    return passages


def format_numbered_passages(passages: list[dict]) -> str:
    """[1] (solar_system.md) text...   - the numbers are what the LLM cites."""
    blocks = []
    for p in passages:
        location = p["source"] + (f", page {p['page']}" if p["page"] else "")
        blocks.append(f"[{p['number']}] ({location})\n{p['text']}")
    return "\n\n".join(blocks)


# ------------------------------------------------------------------
# 5. Answering with numbered citations
# ------------------------------------------------------------------
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are Study Buddy, a friendly study assistant.\n"
     "Answer the student's question using ONLY the numbered passages from their notes below.\n"
     "Rules:\n"
     "- After each fact, cite the passage number in square brackets, e.g. [1] or [1][3].\n"
     f"- If the passages do not contain the answer, reply exactly: \"{NOT_FOUND_MESSAGE}\"\n"
     "- Be clear and concise. Use short paragraphs or bullet points.\n\n"
     "PASSAGES:\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


def build_answer_chain(temperature: float = 0.3):
    """
    prompt | llm   (no parser on purpose!)
    Keeping the AIMessage lets us read token usage (usage_metadata) after streaming.
    """
    return ANSWER_PROMPT | get_chat_model(temperature)


def prepare_answer(vector_store, question: str, history: list | None = None, **search_settings) -> dict:
    """
    Steps 1-4: rewrite the question and retrieve passages.
    Returns everything needed to generate (or stream) the answer afterwards.
    """
    history = history or []
    standalone_question = rewrite_question(question, history)
    passages = retrieve_passages(vector_store, standalone_question, **search_settings)
    return {
        "standalone_question": standalone_question,
        "passages": passages,
        "chain_input": {
            "context": format_numbered_passages(passages),
            "history": history[-HISTORY_WINDOW:],
            "question": standalone_question,
        },
    }


@traceable(name="Ask Notes (conversational)", process_inputs=hide_vector_store)
def ask(vector_store, question: str, history: list | None = None, **search_settings) -> dict:
    """The full pipeline in one call (no streaming). Used by evaluation (step 16) and the demo below."""
    prepared = prepare_answer(vector_store, question, history, **search_settings)

    if not prepared["passages"]:  # nothing relevant -> honest answer, no LLM call needed
        return {"answer": NOT_FOUND_MESSAGE, **prepared}

    message = build_answer_chain().invoke(prepared["chain_input"], config={"run_name": "Generate Answer"})
    return {"answer": message.text, "usage": message.usage_metadata, **prepared}


def to_history(chat: list[dict]) -> list:
    """Convert [{"role": "user"/"assistant", "content": ...}] into LangChain messages (step 02)."""
    return [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in chat]


if __name__ == "__main__":
    vector_store = load_vector_store()

    print("=== 1. Scored retrieval ===")
    for p in retrieve_passages(vector_store, "What is the largest planet?", min_score=0):
        print(f"  [{p['number']}] score {p['score']}  {p['source']}  {p['text'][:60]!r}")

    print("\n=== 2. Filter: search ONLY photosynthesis.txt ===")
    for p in retrieve_passages(vector_store, "energy", sources=["photosynthesis.txt"], min_score=0):
        print(f"  [{p['number']}] {p['source']}  {p['text'][:60]!r}")

    print("\n=== 3. Answer with citations ===")
    first = ask(vector_store, "Tell me about Jupiter.")
    print(first["answer"])

    print("\n=== 4. A follow-up question ===")
    history = [HumanMessage(content="Tell me about Jupiter."), AIMessage(content=first["answer"])]
    follow_up = ask(vector_store, "How many moons does it have?", history)
    print("Rewritten to:", follow_up["standalone_question"])
    print(follow_up["answer"])

    print("\n=== 5. Not in the notes ===")
    print(ask(vector_store, "Who painted the Mona Lisa?")["answer"])
