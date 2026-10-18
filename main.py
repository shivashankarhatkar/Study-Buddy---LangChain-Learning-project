"""
main.py  -  Study Buddy: your AI study assistant
================================================
📖 Read first: learning_notes/15_full_app.md  (simple explanation with diagrams)

This is the finished app. It doesn't introduce anything new - it simply
connects the pieces you built in steps 01-14:

    1. Ask a question about my notes   -> RAG chain with sources   (step 11)
    2. Summarize a notes file          -> summary chain (LCEL)     (steps 05, 07)
    3. Quiz me on a topic              -> RAG + structured output  (steps 04, 11)
    4. Chat with Study Buddy (agent)   -> agent + tools + memory   (steps 12, 13)
    5. Rebuild the notes index         -> load, split, embed, store (steps 07-10)

    Every feature is traced in LangSmith with its own name and tags   (step 14)

Run:   python main.py
"""

from config import print_tracing_status
from step04_output_parsers import answers_match
from step05_chains_lcel import build_summary_chain
from step07_document_loaders import load_notes
from step10_vector_store import build_vector_store
from step11_rag import build_rag_chain_with_sources, build_notes_quiz_chain
from step13_agent import build_study_agent, ask_agent


def trace_config(run_name: str, feature: str) -> dict:
    """
    Build the "config" we pass to .invoke()/.stream() (see step 14).
    It does not change the answer - it only makes LangSmith traces easy to read:
        run_name -> the name shown in LangSmith, e.g. "Ask Notes"
        tags     -> labels to filter by, e.g. show only the "quiz" runs
    """
    return {"run_name": run_name, "tags": ["study-buddy", feature]}


def ask_notes_question():
    """Option 1: answer a question using the notes (RAG)."""
    rag_chain = build_rag_chain_with_sources()
    question = input("\nYour question: ")

    result = rag_chain.invoke(question, config=trace_config("Ask Notes", "rag"))

    print("\nAnswer:\n" + result["answer"])
    sources = sorted({doc.metadata["source"] for doc in result["docs"]})
    print("\nSources searched:", ", ".join(sources))


def summarize_notes_file():
    """Option 2: pick a notes file and get a revision summary (streamed)."""
    documents = load_notes()
    file_names = sorted({doc.metadata["source"] for doc in documents})

    print("\nYour notes files:")
    for number, name in enumerate(file_names, start=1):
        print(f"  {number}. {name}")
    choice = int(input("Pick a file number: "))
    chosen_file = file_names[choice - 1]

    # Join all pages/parts of the chosen file into one text
    notes_text = "\n".join(doc.page_content for doc in documents if doc.metadata["source"] == chosen_file)

    print(f"\nSummary of {chosen_file}:\n")
    # .stream() prints the summary as it is generated
    summary_stream = build_summary_chain().stream(
        {"notes": notes_text},
        config=trace_config("Summarize Notes", "summary"),
    )
    for chunk in summary_stream:
        print(chunk, end="", flush=True)
    print()


def take_quiz():
    """Option 3: generate a multiple-choice quiz from the notes and play it."""
    topic = input("\nQuiz topic (e.g. 'planets', 'python lists'): ")
    print("Creating your quiz...")

    quiz = build_notes_quiz_chain().invoke(
        {"topic": topic, "num_questions": 3},
        config=trace_config("Notes Quiz", "quiz"),
    )

    score = 0
    for number, q in enumerate(quiz.questions, start=1):
        print(f"\nQ{number}. {q.question}")
        letters = ["a", "b", "c", "d"]
        for letter, option in zip(letters, q.options):
            print(f"   {letter}) {option}")

        # A question can have more than one correct answer, e.g. "a,c"
        answer = input("Your answer(s), e.g. b or a,c: ").strip().lower()
        chosen = [q.options[letters.index(letter.strip())]
                  for letter in answer.split(",") if letter.strip() in letters[:len(q.options)]]

        if answers_match(chosen, q.correct_answers):
            score += 1
            print("Correct! " + q.explanation)
        else:
            print(f"Not quite. Correct answer(s): {', '.join(q.correct_answers)}. {q.explanation}")

    print(f"\nYour score: {score}/{len(quiz.questions)}")


def chat_with_agent(agent):
    """Option 4: free chat with the agent (it can search notes, calculate, save flashcards)."""
    print("\nChat with Study Buddy. Try: 'save a flashcard about photosynthesis' or 'show my flashcards'.")
    print("Type 'back' to return to the menu.")
    while True:
        message = input("\nYou: ")
        if message.strip().lower() == "back":
            break
        print("Study Buddy:", ask_agent(agent, message))


def main():
    print("=" * 50)
    print("   STUDY BUDDY  -  your LangChain study assistant")
    print("=" * 50)
    print_tracing_status()

    # Create the agent once, so its memory lasts for the whole session.
    agent = build_study_agent()

    while True:
        print("\nWhat would you like to do?")
        print("  1. Ask a question about my notes")
        print("  2. Summarize a notes file")
        print("  3. Quiz me on a topic")
        print("  4. Chat with Study Buddy (agent)")
        print("  5. Rebuild the notes index (after adding new notes)")
        print("  6. Exit")
        choice = input("> ").strip()

        try:
            if choice == "1":
                ask_notes_question()
            elif choice == "2":
                summarize_notes_file()
            elif choice == "3":
                take_quiz()
            elif choice == "4":
                chat_with_agent(agent)
            elif choice == "5":
                build_vector_store()
            elif choice == "6":
                print("Good luck with your studies!")
                break
            else:
                print("Please choose a number from 1 to 6.")
        except Exception as error:
            # Keep the app running even if one request fails (e.g. network problem)
            print(f"\nSomething went wrong: {error}")


if __name__ == "__main__":
    main()
