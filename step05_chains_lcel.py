"""
STEP 05  -  Chains with LCEL: connecting the pieces with the | pipe
===================================================================
📖 Read first: learning_notes/05_chains_lcel.md  (simple explanation with diagrams)

In step 04 we did this by hand:
    prompt_value = prompt.invoke(...)
    ai_message   = llm.invoke(prompt_value)
    text         = parser.invoke(ai_message)

LCEL ("LangChain Expression Language") lets us write the same thing as:

    chain = prompt | llm | parser
    text  = chain.invoke({...})

The | ("pipe") means: "send the output of the left side into the right side".
Like an assembly line:   dict --> [prompt] --> messages --> [llm] --> AIMessage --> [parser] --> str

Because the whole chain is itself a Runnable, it ALSO gets
.invoke(), .stream() and .batch() for free.

Other useful building blocks shown here:
    RunnableLambda    -> wrap your own Python function so it can be part of a chain
    RunnableParallel  -> run several chains at the same time on the same input
    RunnablePassthrough -> pass the input through unchanged (handy in step 11)

Run this file:   python step05_chains_lcel.py
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

from config import get_chat_model
from step03_prompt_templates import EXPLAIN_PROMPT
from step04_output_parsers import Quiz


# ------------------------------------------------------------------
# Chain 1: Explain a topic   (prompt | llm | parser)
# ------------------------------------------------------------------
def build_explain_chain():
    """Input: {"topic": ..., "level": ...}  ->  Output: explanation string"""
    return EXPLAIN_PROMPT | get_chat_model() | StrOutputParser()


# ------------------------------------------------------------------
# Chain 2: Summarize study notes   (used by the main app)
# ------------------------------------------------------------------
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are Study Buddy. Summarize the student's notes into a revision sheet with:\n"
     "1. A 2-sentence overview\n"
     "2. 5 key points as bullet points\n"
     "3. Important terms with one-line meanings\n"
     "Only use information from the notes."),
    ("human", "Here are my notes:\n\n{notes}"),
])


def build_summary_chain(temperature: float = 0.3):
    """Input: {"notes": "...long text..."}  ->  Output: summary string"""
    return SUMMARY_PROMPT | get_chat_model(temperature) | StrOutputParser()


# ------------------------------------------------------------------
# Chain 3: Generate a quiz   (prompt | structured llm)
# ------------------------------------------------------------------
QUIZ_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are Study Buddy. Write {num_questions} multiple-choice questions to test a "
     "student on the study material below.\n"
     "- Each question has exactly 4 options and ONE OR MORE correct answers. "
     "Make some questions have more than one correct answer.\n"
     "- Difficulty: {difficulty}. (Easy = recall a single fact. Medium = understand and explain. "
     "Hard = combine several facts or apply them.)\n"
     "- Use ONLY facts found in the material.\n\nSTUDY MATERIAL:\n{material}"),
    ("human", "Make the quiz about: {topic}"),
]).partial(difficulty="medium")
# .partial() gives {difficulty} a DEFAULT value. Callers may still pass their own
# "difficulty" (the web app does); callers that don't pass it get "medium".


def build_quiz_chain(temperature: float = 0.3):
    """
    Input:  {"material": ..., "topic": ..., "num_questions": ..., "difficulty" (optional)}
    Output: a Quiz object (see step04)

    Note: no parser needed here - with_structured_output() already returns a Quiz.
    """
    quiz_llm = get_chat_model(temperature).with_structured_output(Quiz)
    return QUIZ_PROMPT | quiz_llm


# ------------------------------------------------------------------
# RunnableLambda: put your own Python function inside a chain
# ------------------------------------------------------------------
def count_words(text: str) -> str:
    """A normal Python function. Adds a word count at the end of the text."""
    return f"{text}\n\n(Word count: {len(text.split())})"


def build_explain_with_wordcount_chain():
    # RunnableLambda turns count_words into a chain step.
    return build_explain_chain() | RunnableLambda(count_words)


# ------------------------------------------------------------------
# RunnableParallel: run several chains on the same input at once
# ------------------------------------------------------------------
def build_study_pack_chain():
    """
    Input:  {"topic": ..., "level": ...}
    Output: {"explanation": "...", "mnemonic": "..."}   (both made in parallel)
    """
    mnemonic_chain = (
        ChatPromptTemplate.from_template("Create one short, catchy mnemonic to remember: {topic}")
        | get_chat_model(temperature=0.8)
        | StrOutputParser()
    )

    # Each key becomes a key in the output dict. Both chains receive the SAME input.
    return RunnableParallel(
        explanation=build_explain_chain(),
        mnemonic=mnemonic_chain,
    )


# ------------------------------------------------------------------
# Chaining chains: the output of one chain feeds the next one
# ------------------------------------------------------------------
def build_explain_then_quiz_chain():
    """
    Step A: explain the topic.
    Step B: turn that explanation into a 2-question quiz.
    Input: {"topic": ..., "level": ...}  ->  Output: Quiz
    """
    explain_chain = build_explain_chain()

    # This small function reshapes the data between the two chains:
    # the quiz chain needs "material", "topic" and "num_questions".
    def prepare_quiz_input(inputs: dict) -> dict:
        explanation = explain_chain.invoke(inputs)
        return {"material": explanation, "topic": inputs["topic"], "num_questions": 2}

    return RunnableLambda(prepare_quiz_input) | build_quiz_chain()


if __name__ == "__main__":
    topic_input = {"topic": "how vaccines work", "level": "beginner"}

    print("=== 1. A basic chain: prompt | llm | parser ===")
    print(build_explain_chain().invoke(topic_input))

    print("\n=== 2. The same chain, streamed ===")
    for chunk in build_explain_chain().stream({"topic": "black holes", "level": "beginner"}):
        print(chunk, end="", flush=True)
    print()

    print("\n=== 3. batch() on a chain ===")
    results = build_explain_chain().batch([
        {"topic": "friction", "level": "beginner"},
        {"topic": "inertia", "level": "beginner"},
    ])
    for result in results:
        print("-", result[:120], "...")

    print("\n=== 4. RunnableLambda (custom function in a chain) ===")
    print(build_explain_with_wordcount_chain().invoke(topic_input))

    print("\n=== 5. RunnableParallel ===")
    pack = build_study_pack_chain().invoke(topic_input)
    print("Explanation:", pack["explanation"])
    print("Mnemonic   :", pack["mnemonic"])

    print("\n=== 6. Chain feeding another chain ===")
    quiz = build_explain_then_quiz_chain().invoke(topic_input)
    for q in quiz.questions:
        print("Q:", q.question, "| Answer(s):", ", ".join(q.correct_answers))
