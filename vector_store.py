import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import subprocess
import json


# ------------------------------------------------
# 🔍 Auto-detect best lightweight model
# ------------------------------------------------
def get_best_model():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            return "qwen:0.5b"  # default fallback

        available_models = result.stdout.lower()

        if "qwen:0.5b" in available_models:
            return "qwen:0.5b"
        elif "gemma:2b" in available_models:
            return "gemma:2b"
        elif "tinyllama" in available_models:
            return "tinyllama:latest"
        else:
            return "qwen:0.5b"  # safe default
    except Exception:
        return "qwen:0.5b"


MODEL_NAME = get_best_model()
st.sidebar.success(f"Using model: {MODEL_NAME}")


# ------------------------------------------------
# 📖 Setup Embeddings & LLM
# ------------------------------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")

llm = Ollama(model=MODEL_NAME, temperature=0.2)

# Memory for multi-turn conversation
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)


# ------------------------------------------------
# 📌 Streamlit UI
# ------------------------------------------------
st.title("💬 Local PDF Chat with Agentic RAG")
st.write("Upload a PDF and ask unlimited questions (multi-turn chat).")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save file temporarily
    pdf_path = os.path.join("temp.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load & index PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    vectorstore = FAISS.from_documents(documents, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    # Chat input
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display old messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Ask something about the PDF..."):
        # Save user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Run agent
        with st.chat_message("assistant"):
            response = qa_chain.run(prompt)
            st.markdown(response)

        # Save assistant reply
        st.session_state.messages.append({"role": "assistant", "content": response})
