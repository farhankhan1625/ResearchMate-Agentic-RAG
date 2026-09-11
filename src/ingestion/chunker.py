from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.loader import load_documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Original pages: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    return chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    print("\n--- First Chunk ---")
    print(chunks[0].page_content)
    print("\nMetadata:")
    print(chunks[0].metadata)