from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader

DATA_DIR = Path("data/documents")


def load_documents():
    loader = PyPDFDirectoryLoader(str(DATA_DIR))
    documents = loader.load()

    print(f"Loaded pages: {len(documents)}")
    return documents


if __name__ == "__main__":
    documents = load_documents()

    if not documents:
        print("No PDF documents found.")
    else:
        first_doc = documents[0]

        print("\n--- First Page ---")
        print("Source:", first_doc.metadata.get("source"))
        print("Page:", first_doc.metadata.get("page"))

        print("\nText Preview:")
        print(first_doc.page_content[:1000])