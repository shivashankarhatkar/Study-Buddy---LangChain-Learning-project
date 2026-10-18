"""
STEP 04  -  Output Parsers & Structured Output: getting clean, usable answers
=============================================================================
📖 Read first: learning_notes/04_output_parsers.md  (simple explanation with diagrams)

PROBLEM:
    The model returns an AIMessage containing free text. But our program often
    needs something more specific:
        - just a plain string                  -> StrOutputParser
        - a Python dict (from JSON)            -> JsonOutputParser
        - a typed Python object (e.g. a Quiz)  -> llm.with_structured_output(...)

    An "output parser" takes the model's message and converts it into the
    shape your code needs.

In this file we call each step BY HAND (prompt -> model -> parser) so you
can see exactly what goes in and out of each piece. In step 05 you'll see how
LangChain lets you connect them with the | pipe in a single line.

Run this file:   python step04_output_parsers.py
"""

from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import get_chat_model


# ------------------------------------------------------------------
# 1. StrOutputParser: AIMessage  ->  plain string
# ------------------------------------------------------------------
def str_parser_demo():
    llm = get_chat_model()
    parser = StrOutputParser()

    ai_message = llm.invoke("Give me a one-line definition of 'evaporation'.")
    text = parser.invoke(ai_message)  # pulls out just the text

    print("Before parsing:", type(ai_message).__name__)  # AIMessage
    print("After parsing :", type(text).__name__)        # str
    print("Value         :", text)


# ------------------------------------------------------------------
# 2. JsonOutputParser: ask for JSON, get a Python dictionary
# ------------------------------------------------------------------
def json_parser_demo():
    llm = get_chat_model()
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You return ONLY valid JSON. No extra words."),
        ("human", "Give 3 key terms about {topic} as JSON with the shape "
                  '{{"topic": "...", "key_terms": [{{"term": "...", "meaning": "..."}}]}}'),
        # Note: {{ and }} are "escaped" braces. They mean a literal { or }
        # in the final prompt, instead of a placeholder.
    ])

    prompt_value = prompt.invoke({"topic": "volcanoes"})
    ai_message = llm.invoke(prompt_value)
    data = parser.invoke(ai_message)  # JSON text -> Python dict

    print("Type:", type(data).__name__)  # dict
    for item in data["key_terms"]:
        print(f"  - {item['term']}: {item['meaning']}")


# ------------------------------------------------------------------
# 3. Structured output with Pydantic  (the recommended way)
# ------------------------------------------------------------------
# We DESCRIBE the shape we want using Pydantic classes. The Field descriptions
# are sent to the model, so write them clearly - they act like instructions!
# The Study Buddy app uses these classes to generate quizzes.

class QuizQuestion(BaseModel):
    """One multiple-choice question (it may have more than one correct answer)."""
    question: str = Field(description="The question text")
    options: list[str] = Field(description="Exactly 4 possible answers")
    correct_answers: list[str] = Field(description="ALL correct options (one or more), each copied exactly from options")
    explanation: str = Field(description="One sentence explaining why these answers are correct")


class Quiz(BaseModel):
    """A short multiple-choice quiz."""
    topic: str = Field(description="The topic of the quiz")
    questions: list[QuizQuestion] = Field(description="The list of quiz questions")


def answers_match(chosen: list[str], correct: list[str]) -> bool:
    """A question is right only if the student picked EXACTLY the correct options
    (ignoring capital letters and extra spaces)."""
    def normalize(options):
        return {option.strip().lower() for option in options}
    return normalize(chosen) == normalize(correct)


# The same idea for flashcards (used by build_flashcards_chain in step 11)
class Flashcard(BaseModel):
    """One revision flashcard."""
    question: str = Field(description="A short question on the front of the card")
    answer: str = Field(description="A short answer (1-2 sentences) on the back of the card")


class FlashcardSet(BaseModel):
    """A set of revision flashcards about one topic."""
    topic: str = Field(description="The topic of the flashcards")
    cards: list[Flashcard] = Field(description="The flashcards")


def structured_output_demo():
    llm = get_chat_model()

    # with_structured_output() returns a NEW model object that always
    # answers with a Quiz object instead of free text.
    # (Under the hood, Gemini is told the exact schema to follow.)
    quiz_llm = llm.with_structured_output(Quiz)

    quiz = quiz_llm.invoke("Create a 2 question quiz about the water cycle for beginners.")

    print("Type:", type(quiz).__name__)  # Quiz  -> a real Python object!
    print("Topic:", quiz.topic)
    for number, q in enumerate(quiz.questions, start=1):
        print(f"\nQ{number}. {q.question}")
        for option in q.options:
            print("    -", option)
        print("  Answer(s):", ", ".join(q.correct_answers))


if __name__ == "__main__":
    print("=== 1. StrOutputParser ===")
    str_parser_demo()

    print("\n=== 2. JsonOutputParser ===")
    json_parser_demo()

    print("\n=== 3. Structured output (Pydantic) ===")
    structured_output_demo()
