"""
STEP 12  -  Tools: letting the LLM use your Python functions
============================================================
📖 Read first: learning_notes/12_tools.md  (simple explanation with diagrams)

An LLM can only produce text. It cannot search files, do exact maths or
save data. TOOLS fix this: you give the model a list of Python functions,
and the model can ASK you to run one of them.

HOW IT WORKS (important!):
    1. You describe your tools to the model         -> llm.bind_tools(tools)
    2. The model decides a tool is needed and replies with a "tool call":
           {"name": "calculate", "args": {"a": 12, "b": 7, "operation": "multiply"}}
       The model does NOT run anything itself!
    3. YOUR code runs the function and sends the result back as a ToolMessage.
    4. The model reads the result and writes the final answer.

THE @tool DECORATOR
    Putting @tool above a function turns it into a LangChain tool. LangChain reads:
        - the function NAME         -> the tool's name
        - the DOCSTRING             -> the description (the model reads this to
                                       decide WHEN to use the tool, so write it well!)
        - the TYPE HINTS            -> what arguments the tool expects

In step 13 an AGENT will run steps 2-4 automatically in a loop.

Run this file:   python step12_tools.py
"""

import os

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from config import get_chat_model, FLASHCARDS_FILE
from step11_rag import get_retriever, format_docs


# ------------------------------------------------------------------
# Our Study Buddy tools
# ------------------------------------------------------------------
@tool
def search_my_notes(query: str) -> str:
    """Search the student's personal study notes and return the most relevant parts.
    Use this whenever the student asks about something that might be in their notes."""
    docs = get_retriever(k=4).invoke(query)
    return format_docs(docs)


@tool
def calculate(a: float, b: float, operation: str) -> float:
    """Do exact maths on two numbers.
    operation must be one of: 'add', 'subtract', 'multiply', 'divide', 'power'."""
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        return a / b
    if operation == "power":
        return a ** b
    return f"Unknown operation '{operation}'"


@tool
def save_flashcard(question: str, answer: str) -> str:
    """Save a revision flashcard (a question and its answer) for the student to review later."""
    os.makedirs(os.path.dirname(FLASHCARDS_FILE), exist_ok=True)
    with open(FLASHCARDS_FILE, "a", encoding="utf-8") as file:
        file.write(f"Q: {question}\nA: {answer}\n\n")
    return "Flashcard saved."


@tool
def show_flashcards() -> str:
    """Show all the flashcards the student has saved so far."""
    if not os.path.exists(FLASHCARDS_FILE):
        return "No flashcards saved yet."
    with open(FLASHCARDS_FILE, encoding="utf-8") as file:
        return file.read()


def read_flashcards() -> list[dict]:
    """
    Normal Python helper (NOT a tool): read the flashcards file as a list of
    {"question": ..., "answer": ...} dicts, so the web app can display them as cards.
    """
    if not os.path.exists(FLASHCARDS_FILE):
        return []
    cards = []
    with open(FLASHCARDS_FILE, encoding="utf-8") as file:
        for line in file:
            if line.startswith("Q: "):
                cards.append({"question": line[3:].strip(), "answer": ""})
            elif line.startswith("A: ") and cards:
                cards[-1]["answer"] = line[3:].strip()
            elif line.strip() and cards:  # an answer that continues on the next line
                cards[-1]["answer"] += " " + line.strip()
    return cards


# A list of all tools, and a name -> tool lookup (used to run a tool by name)
STUDY_TOOLS = [search_my_notes, calculate, save_flashcard, show_flashcards]
TOOLS_BY_NAME = {t.name: t for t in STUDY_TOOLS}


# ------------------------------------------------------------------
# Demo 1: what does LangChain know about a tool?
# ------------------------------------------------------------------
def inspect_tool_demo():
    print("Name       :", calculate.name)
    print("Description:", calculate.description)
    print("Arguments  :", calculate.args)
    # A tool is a Runnable too, so you can call it directly with .invoke()
    print("Run it     :", calculate.invoke({"a": 6, "b": 7, "operation": "multiply"}))


# ------------------------------------------------------------------
# Demo 2: the full tool-calling cycle, done BY HAND
# ------------------------------------------------------------------
def manual_tool_calling_demo(question: str):
    # bind_tools() tells the model which tools exist (name, description, arguments).
    llm_with_tools = get_chat_model().bind_tools(STUDY_TOOLS)

    messages = [HumanMessage(content=question)]

    # 1) The model decides what to do.
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)

    print("Tool calls requested by the model:")
    for tool_call in ai_message.tool_calls:
        print(f"  -> {tool_call['name']}({tool_call['args']})")

    # 2) WE run each requested tool. Invoking a tool with the tool_call itself
    #    returns a ToolMessage, which links the result to the request (by its id).
    for tool_call in ai_message.tool_calls:
        selected_tool = TOOLS_BY_NAME[tool_call["name"]]
        tool_message = selected_tool.invoke(tool_call)
        print(f"  <- result: {str(tool_message.content)[:80]}")
        messages.append(tool_message)

    # 3) Send everything back so the model can write the final answer.
    final_message = llm_with_tools.invoke(messages)
    print("\nFinal answer:", final_message.text)


if __name__ == "__main__":
    print("=== 1. Inspecting a tool ===")
    inspect_tool_demo()

    print("\n=== 2. Manual tool calling ===")
    manual_tool_calling_demo("What is 17.5 multiplied by 4.2?")
