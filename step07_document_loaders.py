"""
STEP 07  -  Document Loaders: reading your notes into LangChain
===============================================================
📖 Read first: learning_notes/07_document_loaders.md  (simple explanation with diagrams)

From here on we build "RAG" (Retrieval Augmented Generation) step by step:
the ability to answer questions from YOUR OWN notes, which the LLM has never seen.

    step 07  Load      -> read files into Document objects        <- you are here
    step 08  Split     -> cut documents into small chunks
    step 09  Embed     -> turn each chunk into a vector (numbers)
    step 10  Store     -> save vectors in a vector store
    step 11  Retrieve  -> find the chunks relevant to a question, give them to the LLM

WHAT IS A DOCUMENT?
    LangChain stores text in a "Document" object with two fields:
        page_content -> the actual text
        metadata     -> extra info about the text (file name, page number...)
    Metadata is very useful later: it lets the app say WHERE an answer came from.

WHAT IS A DOCUMENT LOADER?
    A class that reads some source (files, websites, databases...) and turns it into
    a list of Documents. Every loader has a .load() method.
    LangChain has hundreds of ready-made loaders (PDFs, web pages, Notion, YouTube...).
    Here we write our OWN small loader by extending LangChain's BaseLoader, so you
    can see that there is no magic: a loader just reads files and creates Documents.

Run this file:   python step07_document_loaders.py
"""

from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

from config import NOTES_FOLDER


class StudyNotesLoader(BaseLoader):
    """
    Loads every .txt, .md and .pdf file from a folder (or one single file).

    - Text/markdown files -> one Document per file
    - PDF files           -> one Document per page (so we know the page number)

    We only have to write lazy_load(). BaseLoader gives us .load() for free,
    which simply collects everything lazy_load() yields into a list.
    """

    def __init__(self, path: str):
        self.path = Path(path)

    def lazy_load(self):
        # A single file -> just that file. A folder -> every file in it (sorted for a predictable order).
        files = [self.path] if self.path.is_file() else sorted(self.path.iterdir())

        for file_path in files:
            suffix = file_path.suffix.lower()

            if suffix in (".txt", ".md"):
                text = file_path.read_text(encoding="utf-8")
                # "yield" hands back one Document at a time
                yield Document(
                    page_content=text,
                    metadata={"source": file_path.name},
                )

            elif suffix == ".pdf":
                reader = PdfReader(file_path)
                for page_number, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():  # skip empty pages
                        yield Document(
                            page_content=text,
                            metadata={"source": file_path.name, "page": page_number},
                        )


def load_notes(folder: str = NOTES_FOLDER) -> list[Document]:
    """Load all notes from the notes folder as a list of Documents."""
    loader = StudyNotesLoader(folder)
    return loader.load()


def load_file(file_path: str) -> list[Document]:
    """Load ONE file (used when a new file is uploaded in the web app)."""
    return StudyNotesLoader(file_path).load()


if __name__ == "__main__":
    documents = load_notes()
    print(f"Loaded {len(documents)} documents from '{NOTES_FOLDER}/'\n")

    for doc in documents:
        print("Metadata     :", doc.metadata)
        print("Length       :", len(doc.page_content), "characters")
        print("First 150 ch.:", doc.page_content[:150].replace("\n", " "), "...\n")

    # Tip: with the "langchain-community" package you could also use ready-made
    # loaders such as TextLoader("file.txt") or PyPDFLoader("file.pdf").
    # They work exactly like ours: loader.load() -> list[Document].
