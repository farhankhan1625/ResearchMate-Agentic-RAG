from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Same embedding model used during vector-store creation
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load existing ChromaDB
vectorstore = Chroma(
    persist_directory="data/chroma_db",
    embedding_function=embeddings
)


def retrieve_documents(query, k=3, max_distance=1.5):
    """Retrieve relevant document chunks using a distance threshold."""

    results = vectorstore.similarity_search_with_score(
        query,
        k=k
    )

    docs = [
        doc
        for doc, score in results
        if score <= max_distance
    ]

    return docs


if __name__ == "__main__":

    query = "What is the Transformer architecture?"

    results = vectorstore.similarity_search_with_score(query, k=3)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} documents:\n")

    for i, (doc, score) in enumerate(results, 1):
        print(f"--- Document {i} ---")
        print(f"Distance Score: {score:.4f}")
        print(doc.page_content[:500])
        print("Page:", doc.metadata.get("page_label", "Unknown"))
        print()