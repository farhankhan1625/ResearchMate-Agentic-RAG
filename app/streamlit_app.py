import sys
from pathlib import Path

import streamlit as st

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.rag import answer_question
from src.retrieval.vector_store import create_vector_store
# --------------------------------------------------
# CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="ResearchMate",
    page_icon="📚",
    layout="wide"
)

DATA_DIR = Path("data/documents")
DB_DIR = Path("data/chroma_db")

DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📚 ResearchMate")
st.subheader("Local AI Research Assistant")
st.write(
    "Upload research documents and ask questions using "
    "retrieval-augmented generation."
)


# --------------------------------------------------
# SIDEBAR - DOCUMENT UPLOAD
# --------------------------------------------------

with st.sidebar:

    st.header("📄 Documents")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.button("🔄 Index Document", use_container_width=True):

            try:
                # Save uploaded PDF
                file_path = DATA_DIR / uploaded_file.name

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Rebuild vector database
             

                with st.spinner("Indexing document..."):

                    create_vector_store()

                st.success(
                    f"Indexed: {uploaded_file.name}"
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Indexing failed: {str(e)}"
                )


    st.divider()

    st.caption(
        "Embeddings: all-MiniLM-L6-v2"
    )

    st.caption(
        "LLM: Llama 3.2 via Ollama"
    )

    st.caption(
        "Vector DB: ChromaDB"
    )


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your research document..."
)


if question:

    # User message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("ResearchMate is thinking..."):

            try:

                answer, docs = answer_question(question)

                st.markdown(answer)

                # Sources
                if docs:

                    st.markdown("### 📖 Sources")

                    for i, doc in enumerate(docs, 1):

                        source = Path(
                            doc.metadata.get(
                                "source",
                                "Unknown"
                            )
                        ).name

                        page = doc.metadata.get(
                            "page_label",
                            doc.metadata.get(
                                "page",
                                "Unknown"
                            )
                        )

                        st.write(
                            f"**{i}.** `{source}` — "
                            f"Page {page}"
                        )

            except Exception as e:

                answer = (
                    "⚠️ Unable to generate an answer. "
                    "Please make sure Ollama is running "
                    "and the document has been indexed."
                )

                st.error(str(e))


    # Save assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )