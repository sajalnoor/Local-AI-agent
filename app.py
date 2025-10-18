# app.py
import json
import re
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun

# ----------------------------
# Config
# ----------------------------
MODEL_NAME = "tinyllama:latest"  # change if you want another ollama model
PDF_TEMP_PATH = "temp.pdf"
PDF_SNIPPET_CHAR_LIMIT = 3000  # limit how much PDF text we send to the LLM to avoid context overflow

st.sidebar.success(f"Using model: {MODEL_NAME}")
st.title("🤖 Agentic RAG — PDF + Web")

# ----------------------------
# Initialize LLMs & tools
# ----------------------------
# Decision LLM (deterministic)
llm_decider = Ollama(model=MODEL_NAME, temperature=0.0)

# Answer LLM (a bit creative is OK)
llm_answer = Ollama(model=MODEL_NAME, temperature=0.2)

# Local embeddings (no API keys)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Web search tool
web_search = DuckDuckGoSearchRun()


# ----------------------------
# Helpers
# ----------------------------
def safe_parse_json_from_text(text: str):
    """
    Attempt to pull JSON substring from text and parse it.
    Returns dict or None.
    """
    if not text:
        return None
    # find first { ... } block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        # try simple tokens: pdf / web / both
        txt = text.strip().lower()
        if "pdf" in txt and "web" not in txt:
            return {"action": "pdf"}
        if "web" in txt and "pdf" not in txt:
            return {"action": "web"}
        if "both" in txt:
            return {"action": "both"}
        return None
    json_str = m.group(0)
    try:
        return json.loads(json_str)
    except Exception:
        # try to fix single quotes
        try:
            return json.loads(json_str.replace("'", '"'))
        except Exception:
            return None


def run_pdf_retriever_text(retriever, query: str, k=3) -> (str, list):
    """
    Returns (text_summary, docs_list)
    text_summary is a plain-text concatenation suitable to send to the LLM.
    docs_list is the list of Document objects (kept for showing sources).
    """
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "", []
    pieces = []
    for d in docs[:k]:
        source = d.metadata.get("source", PDF_TEMP_PATH)
        page = d.metadata.get("page", "Unknown")
        snippet = d.page_content
        # trim long snippets
        if len(snippet) > PDF_SNIPPET_CHAR_LIMIT:
            snippet = snippet[:PDF_SNIPPET_CHAR_LIMIT] + "\n...[truncated]"
        pieces.append(f"Source: {source} | Page: {page}\n{snippet}")
    text = "\n\n---\n\n".join(pieces)
    return text, docs


def call_decision_llm(prompt: str) -> str:
    """
    Asks the LLM to choose an action. Returns raw LLM text.
    The prompt forces a strict JSON response: {"action":"pdf"|"web"|"both"}
    """
    decision_prompt = (
        "Decide whether the user's question should be answered from the uploaded PDF, from the web, or both.\n\n"
        "Respond ONLY with a JSON object and nothing else, e.g. {\"action\":\"pdf\"} or {\"action\":\"web\"} or {\"action\":\"both\"}.\n\n"
        "Choose 'pdf' when the question likely can be answered using the uploaded PDF alone.\n"
        "Choose 'web' when the PDF probably doesn't contain the information and an up-to-date web search is required.\n"
        "Choose 'both' when both the PDF and web would be helpful.\n\n"
        f"User question: {prompt}\n\n"
    )
    return llm_decider.invoke(decision_prompt)


def synthesize_answer_from_pdf(context_text: str, question: str) -> str:
    prompt = (
        "You are an assistant. Use ONLY the PDF context below to answer the question. "
        "If the PDF doesn't contain the answer, respond with exactly: NOT_IN_PDF\n\n"
        f"PDF CONTEXT:\n{context_text}\n\nQUESTION: {question}\n\n"
        "Provide a concise answer and cite PDF pages where possible using [PDF:page]."
    )
    return llm_answer.invoke(prompt).strip()


def synthesize_answer_from_web(web_text: str, question: str) -> str:
    prompt = (
        "You are an assistant. Use ONLY the web search results below to answer the question. "
        "Write a concise answer and indicate that the info came from the web.\n\n"
        f"WEB SEARCH RESULTS:\n{web_text}\n\nQUESTION: {question}\n\n"
        "Provide a concise answer and (optionally) short summary of the sources."
    )
    return llm_answer.invoke(prompt).strip()


def synthesize_answer_from_both(pdf_text: str, web_text: str, question: str) -> str:
    prompt = (
        "You are an assistant. Use the PDF context and the web results below. "
        "Synthesize a final concise answer. When you cite PDF content, use [PDF:page]. "
        "When you reference web facts, mark them as (web).\n\n"
        f"PDF CONTEXT:\n{pdf_text}\n\nWEB RESULTS:\n{web_text}\n\nQUESTION: {question}\n\n"
        "Provide a short synthesized answer that clearly indicates which info came from the PDF and which from the web."
    )
    return llm_answer.invoke(prompt).strip()


# ----------------------------
# UI: upload and index PDF
# ----------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
retriever = None
indexed_docs = None

if uploaded_file:
    with open(PDF_TEMP_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Saved uploaded PDF to temp.pdf")

    loader = PyPDFLoader(PDF_TEMP_PATH)
    raw_pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(raw_pages)
    indexed_docs = docs

    # build FAISS vectorstore
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    st.success(f"Indexed {len(docs)} chunks from the PDF.")


# ----------------------------
# Chat area
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role":..., "text":...}

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

user_input = st.chat_input("Ask anything (agentic RAG):")
if user_input:
    st.session_state.history.append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---------- Decision step ----------
    decision_raw = call_decision_llm(user_input)
    parsed = safe_parse_json_from_text(decision_raw)
    action = (parsed or {}).get("action") if parsed else None

    # fallback heuristics
    if action not in {"pdf", "web", "both"}:
        # if we have retriever and the top chunks look non-empty, prefer pdf; else web
        if retriever:
            sample_docs = retriever.get_relevant_documents(user_input)
            if sample_docs and any(d.page_content.strip() for d in sample_docs):
                action = "pdf"
            else:
                action = "web"
        else:
            action = "web"

    # ---------- Execute tool(s) ----------
    pdf_text, pdf_docs = ("", [])
    web_text = ""

    if action in ("pdf", "both") and retriever:
        pdf_text, pdf_docs = run_pdf_retriever_text(retriever, user_input, k=4)

    if action in ("web", "both"):
        # DuckDuckGo returns a text summary/listing
        web_text = web_search.run(user_input)

    # ---------- Synthesis step ----------
    final_answer = ""
    try:
        if action == "pdf":
            final_answer = synthesize_answer_from_pdf(pdf_text, user_input)
            # if model says NOT_IN_PDF, do a web fallback automatically
            if final_answer.strip().upper() == "NOT_IN_PDF":
                # run web and synthesize
                web_text = web_search.run(user_input)
                final_answer = synthesize_answer_from_web(web_text, user_input)
        elif action == "web":
            final_answer = synthesize_answer_from_web(web_text, user_input)
        else:  # both
            final_answer = synthesize_answer_from_both(pdf_text, web_text, user_input)
    except Exception as e:
        final_answer = f"⚠️ Error while synthesizing answer: {e}"

    # show answer
    with st.chat_message("assistant"):
        st.markdown(final_answer)

        # show PDF sources if used
        if pdf_docs:
            with st.expander("PDF sources (click to open file)"):
                st.markdown(f"[Open uploaded PDF]({PDF_TEMP_PATH})")
                for d in pdf_docs:
                    page = d.metadata.get("page", "Unknown")
                    source = d.metadata.get("source", PDF_TEMP_PATH)
                    st.markdown(f"- Page {page} — {source}")

        # optionally show web snippet
        if web_text:
            with st.expander("Web results (summary)"):
                st.code(web_text[:4000])  # show first 4k chars to keep UI tidy

    st.session_state.history.append({"role": "assistant", "text": final_answer})
