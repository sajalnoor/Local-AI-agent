# 🤖 Agentic RAG — Local PDF + Web Q&A using Ollama

A lightweight **Agentic Retrieval-Augmented Generation (RAG)** system that runs **completely offline** using local Ollama models — with optional online web search via DuckDuckGo.  
It can answer questions from uploaded PDFs or synthesize answers from both **PDF + Web** intelligently.

---

## 🧠 Features
- **Offline AI** — Runs locally using Ollama (no APIs or keys needed)
- **Agentic Decision-Making** — Model decides whether to retrieve from PDF, Web, or both
- **Multi-turn Chat Support**
- **FAISS-based Retrieval** for fast context lookups
- **Streamlit Interface** for user-friendly interaction
- **Privacy-Preserving** — No data leaves your machine

---

## 🧩 Project Files

| File | Description |
|------|--------------|
| `main.py` | Command-line based RAG Q&A from PDFs |
| `vector_store.py` | Streamlit PDF Chat (multi-turn, offline) |
| `app.py` | Full Agentic RAG — combines PDF + Web retrieval |

---

## 🧱 Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- At least one local model pulled (recommended: `qwen:0.5b`, `tinyllama:latest`, or `gemma:2b`)

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<yourusername>/agentic-rag.git
cd agentic-rag

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt
