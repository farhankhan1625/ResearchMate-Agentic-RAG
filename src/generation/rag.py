from src.retrieval.retriever import retrieve_documents
from langchain_ollama import ChatOllama


# --------------------------------------------------
# LOCAL LLM
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# --------------------------------------------------
# RAG ANSWER FUNCTION
# --------------------------------------------------

def answer_question(query, k=10):

    # Retrieve relevant document chunks
    docs = retrieve_documents(
        query,
        k=k
    )

    # Relevance guard
    if not docs:
        return (
            "I could not find relevant information in the provided document.",
            []
        )

    # Combine retrieved chunks
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # Prompt
    prompt = f"""
You are ResearchMate, a research document assistant.

Answer the user's question using ONLY the information
contained in the provided document context.

Rules:
Rules:
1. Answer using only the provided document context.
2. Do not use outside knowledge.
3. Do not invent facts or numerical values.
4. You may calculate derived values when the required numerical values are present in the document context.
5. If the question asks for an average, total, difference, percentage, or similar calculation, perform the calculation using the relevant values from the context.
6. Show the calculation briefly and give the final result with units.
7. If the required information is genuinely missing from the context, say:
   "I could not find the answer in the provided document."
8. Give a concise and direct answer.
Document Context:
-----------------
{context}
-----------------

User Question:
{query}

Answer:
"""

    # Generate answer using local Llama 3.2
    response = llm.invoke(prompt)

    answer = response.content

    return answer, docs


# --------------------------------------------------
# CLI TEST
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("RESEARCHMATE - LOCAL RAG ASSISTANT")
    print("=" * 60)

    while True:

        query = input("\nYou: ")

        if query.lower() in ["exit", "quit"]:

            print("\nResearchMate: Goodbye!")
            break

        answer, docs = answer_question(query)

        print("\nResearchMate:")
        print(answer)

        print("\nSources:")

        if docs:

            for i, doc in enumerate(docs, 1):

                source = doc.metadata.get(
                    "source",
                    "Unknown"
                )

                page = doc.metadata.get(
                    "page_label",
                    doc.metadata.get(
                        "page",
                        "Unknown"
                    )
                )

                print(
                    f"{i}. {source} | Page: {page}"
                )

        else:

            print("No relevant sources found.")

        print("\n" + "-" * 60)