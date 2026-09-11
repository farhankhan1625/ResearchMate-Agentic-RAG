import streamlit as st

from src.generation.rag import answer_question


st.set_page_config(
    page_title="ResearchMate",
    page_icon="📚",
    layout="wide"
)


st.title("📚 ResearchMate")
st.subheader("Local AI Research Assistant")

st.write(
    "Ask questions about your indexed research documents."
)


# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
question = st.chat_input(
    "Ask a question about your document..."
)


if question:

    # Show user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("ResearchMate is thinking..."):

            answer, docs = answer_question(question)

        st.markdown(answer)


        # Show sources
        if docs:

            st.markdown("### 📖 Sources")

            for i, doc in enumerate(docs, 1):

                source = doc.metadata.get(
                    "source",
                    "Unknown"
                )

                page = doc.metadata.get(
                    "page_label",
                    "Unknown"
                )

                st.write(
                    f"**{i}.** {source} — Page {page}"
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )