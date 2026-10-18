"""
STEP 11  -  Retrievers & RAG: answering questions from YOUR notes
=================================================================
📖 Read first: learning_notes/11_retrievers_rag.md  (simple explanation with diagrams)

RAG = Retrieval Augmented Generation
    Retrieval   -> find the chunks of your notes related to the question
    Augmented   -> paste those chunks into the prompt as "context"
    Generation  -> the LLM writes an answer using that context

    The LLM was never trained on your notes, but with RAG it can still answer
    from them, and we can tell it to say "I don't know" when the notes don't
    contain the answer (this reduces made-up answers, called "hallucinations").

WHAT IS A RETRIEVER?
    A Runnable that takes a question (string) and returns a list of relevant Documents.
    vector_store.as_retriever() turns our vector store into one.
    Because a retriever is a Runnable, it can be used inside an LCEL chain with |.

THE FLOW OF OUR RAG CHAIN:
    question
       |-> retriever -> docs -> format_docs -> context
       |-> (question passed through unchanged)
                 \\-> RAG_PROMPT(context, question) -> llm -> parser -> answer

Run this file:   python step11_rag.py
"""

from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from config import get_chat_model
from step04_output_parsers import FlashcardSet
from step05_chains_lcel import build_quiz_chain
from step10_vector_store import load_vector_store


def get_retriever(k: int = 4):
    """Turn the vector store into a retriever that returns the top-k chunks."""
    return load_vector_store().as_retriever(search_kwargs={"k": k})


def format_docs(docs: list[Document]) -> str:
    """Join retrieved chunks into one block of text, labelled with their source file."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source')}]\n{doc.page_content}" for doc in docs
    )


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are Study Buddy. Answer the student's question using ONLY the notes below.\n"
     "If the answer is not in the notes, say: \"I couldn't find that in your notes.\"\n"
     "Keep the answer clear and short, and mention which source file you used.\n\n"
     "NOTES:\n{context}"),
    ("human", "{question}"),
])


# ------------------------------------------------------------------
# Version 1: the classic, minimal RAG chain   (input: question string)
# ------------------------------------------------------------------
def build_simple_rag_chain():
    """Input: "question"  ->  Output: answer string"""
    retriever = get_retriever()

    return (
        # This dict runs both parts on the same input (the question string):
        #   "context":  question -> retriever -> docs -> format_docs -> text
        #   "question": RunnablePassthrough() just passes the question along unchanged
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | get_chat_model()
        | StrOutputParser()
    )


# ------------------------------------------------------------------
# Version 2: RAG that ALSO returns the source documents (used by the app)
# ------------------------------------------------------------------
def build_rag_chain_with_sources():
    """
    Input:  "question"
    Output: {"question": ..., "docs": [...], "context": ..., "answer": ...}

    RunnablePassthrough.assign(key=...) keeps everything already in the dict
    and ADDS a new key. So the dict grows step by step as it moves along.
    """
    retriever = get_retriever()
    answer_chain = RAG_PROMPT | get_chat_model() | StrOutputParser()

    return (
        RunnableParallel(docs=retriever, question=RunnablePassthrough())     # {docs, question}
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))  # + context
        | RunnablePassthrough.assign(answer=answer_chain)                       # + answer
    )


# ------------------------------------------------------------------
# Quiz from notes: RAG + the quiz chain from step 05
# ------------------------------------------------------------------
def build_notes_quiz_chain():
    """
    Input:  {"topic": ..., "num_questions": ...}
    Output: Quiz object, with questions based on the student's own notes.
    """
    retriever = get_retriever(k=6)

    return (
        # itemgetter("topic") picks the topic out of the input dict, so the
        # retriever searches the notes for that topic. The result becomes "material".
        RunnablePassthrough.assign(material=itemgetter("topic") | retriever | format_docs)
        | build_quiz_chain()
    )


# ------------------------------------------------------------------
# Flashcards from notes: the SAME pattern as the quiz, with a different schema
# ------------------------------------------------------------------
FLASHCARDS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are Study Buddy. Create {num_cards} revision flashcards from the study material below. "
     "Each card tests ONE fact. Use ONLY facts found in the material.\n\nSTUDY MATERIAL:\n{material}"),
    ("human", "Make flashcards about: {topic}"),
])


def build_flashcards_chain():
    """Input: {"material": ..., "topic": ..., "num_cards": ...}  ->  Output: FlashcardSet (see step04)"""
    return FLASHCARDS_PROMPT | get_chat_model().with_structured_output(FlashcardSet)


def build_notes_flashcards_chain():
    """
    Input:  {"topic": ..., "num_cards": ...}
    Output: FlashcardSet, with cards based on the student's own notes.
    """
    retriever = get_retriever(k=6)
    return (
        RunnablePassthrough.assign(material=itemgetter("topic") | retriever | format_docs)
        | build_flashcards_chain()
    )


if __name__ == "__main__":
    print("=== 1. What does the retriever return? ===")
    for doc in get_retriever(k=2).invoke("What is a Python list?"):
        print(f"[{doc.metadata['source']}] {doc.page_content[:100]!r}")

    print("\n=== 2. Simple RAG chain ===")
    print(build_simple_rag_chain().invoke("Why is Venus hotter than Mercury?"))

    print("\n=== 3. RAG chain with sources ===")
    result = build_rag_chain_with_sources().invoke("What are the products of photosynthesis?")
    print("Answer :", result["answer"])
    print("Sources:", {doc.metadata["source"] for doc in result["docs"]})

    print("\n=== 4. A question NOT in the notes ===")
    print(build_simple_rag_chain().invoke("Who won the 2010 football World Cup?"))

    print("\n=== 5. Quiz from notes ===")
    quiz = build_notes_quiz_chain().invoke({"topic": "the planets", "num_questions": 2})
    for q in quiz.questions:
        print("Q:", q.question, "| Answer(s):", ", ".join(q.correct_answers))
