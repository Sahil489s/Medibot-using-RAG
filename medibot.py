import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🩺",
    layout="wide",
)


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}

.title {
    text-align:center;
    color:#2E86C1;
    font-size:40px;
    font-weight:bold;
}

.subtitle {
    text-align:center;
    color:gray;
    margin-bottom:20px;
}

.footer {
    text-align:center;
    color:gray;
    font-size:12px;
    margin-top:30px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- VECTOR STORE ----------------
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


# ---------------- MAIN APP ----------------
def main():

    # Sidebar
    with st.sidebar:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/2966/2966486.png",
            width=120
        )

        st.title("⚙️ Settings")

        st.markdown("---")

        st.markdown("""
        ### About

        This AI Medical Assistant uses:

        - LangChain
        - FAISS Vector DB
        - Groq LLM
        - RAG Architecture

        Ask questions related to your medical documents.
        """)

        st.markdown("---")

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Header
    st.markdown(
        '<p class="title">🩺 AI Medical Assistant</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Ask questions from your medical knowledge base</p>',
        unsafe_allow_html=True
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    prompt = st.chat_input("💬 Ask your medical question...")

    if prompt:

        # User Message
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        try:

            with st.spinner("🔍 Searching medical knowledge base..."):

                vectorstore = get_vectorstore()

                GROQ_API_KEY = os.getenv("GROQ_API_KEY")

                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0.3,
                    max_tokens=512,
                    api_key=GROQ_API_KEY
                )

                retrieval_prompt = hub.pull(
                    "langchain-ai/retrieval-qa-chat"
                )

                combine_docs_chain = create_stuff_documents_chain(
                    llm,
                    retrieval_prompt
                )

                rag_chain = create_retrieval_chain(
                    vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    ),
                    combine_docs_chain
                )

                response = rag_chain.invoke(
                    {"input": prompt}
                )

                answer = response["answer"]

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    st.markdown(
        """
        <div class="footer">
        🚀 AI Medical Assistant • Powered by RAG Technology • Developed by Sahil Sharma.
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()