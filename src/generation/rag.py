from langchain_ollama import ChatOllama

from src.retrieval.retriever import retrieve_documents


llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


def answer_question(query):
    docs = retrieve_documents(query, k=3)

    if not docs:
        return (
            "I could not find relevant information in the provided document.",
            []
        )

    context_parts = []

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page_label", "Unknown")

        context_parts.append(
            f"[Source {i} | Page {page}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are ResearchMate, a research assistant.

Answer the question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. If the answer is not supported by the context, say:
"I could not find the answer in the provided document."
3. Keep the answer concise and factual.
4. Mention the relevant source/page when possible.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content, docs


if __name__ == "__main__":

    print("=" * 60)
    print("RESEARCHMATE - LOCAL RAG ASSISTANT")
    print("=" * 60)
    print("Ask questions about your documents.")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        if not question.strip():
            continue

        answer, docs = answer_question(question)

        print("\nResearchMate:")
        print(answer)

        print("\nSources:")

        for i, doc in enumerate(docs, 1):
            print(
                f"{i}. "
                f"{doc.metadata.get('source', 'Unknown')} | "
                f"Page: {doc.metadata.get('page_label', 'Unknown')}"
            )

        print("\n" + "-" * 60 + "\n")
        