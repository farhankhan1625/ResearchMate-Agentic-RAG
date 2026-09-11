from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.ingestion.loader import load_documents
from src.ingestion.chunker import split_documents


DB_DIR = "data/chroma_db"


def create_vector_store():
    print("Loading documents...")
    documents = load_documents()

    print("Creating chunks...")
    chunks = split_documents(documents)

    print(f"Creating embeddings for {len(chunks)} chunks...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print("\nVector store created successfully!")
    print(f"Stored chunks: {len(chunks)}")
    print(f"Database: {DB_DIR}")

    return vector_store


if __name__ == "__main__":
    create_vector_store()