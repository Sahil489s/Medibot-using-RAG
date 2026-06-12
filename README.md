# 🩺 Medibot using RAG

<div align="center">

### AI-Powered Medical Chatbot using Retrieval-Augmented Generation (RAG)

Ask medical questions and receive context-aware answers grounded in a trusted medical knowledge base using **LangChain**, **FAISS**, **HuggingFace Embeddings**, and **Groq LLMs**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

## 📖 Overview

Medibot is a Retrieval-Augmented Generation (RAG) based Medical Assistant that answers medical questions using information retrieved from a curated medical knowledge base.

Instead of relying solely on an LLM's internal knowledge, the chatbot first retrieves relevant medical passages from a FAISS vector database and then generates responses grounded in those sources, reducing hallucinations and improving reliability.

---

## ✨ Features

* 🔍 Semantic Search with FAISS
* 🧠 Retrieval-Augmented Generation (RAG)
* 📄 PDF-based Medical Knowledge Base
* 🤖 Groq LLM Integration (Llama Models)
* 🤗 HuggingFace Embeddings
* 💬 Interactive Streamlit Interface
* 📚 Source Document Retrieval
* ⚡ Fast and Scalable Search
* 🔄 Modular and Extensible Architecture

---

## 🏗️ Architecture

```text
Medical PDF Documents
          │
          ▼
Text Chunking
          │
          ▼
Embedding Generation
          │
          ▼
FAISS Vector Store
          │
          ▼
User Query
          │
          ▼
Retriever (Top-K Search)
          │
          ▼
Prompt Construction
          │
          ▼
Groq / HuggingFace LLM
          │
          ▼
Generated Answer
          │
          ▼
Source References
```

---

## 🚀 Tech Stack

| Category               | Technology            |
| ---------------------- | --------------------- |
| Programming Language   | Python                |
| Framework              | LangChain             |
| Vector Database        | FAISS                 |
| Embeddings             | Sentence Transformers |
| Embedding Model        | all-MiniLM-L6-v2      |
| LLM Provider           | Groq                  |
| Alternative LLM        | HuggingFace           |
| Frontend               | Streamlit             |
| Document Loader        | PyPDF                 |
| Environment Management | Dotenv                |

---

## 📂 Project Structure

```text
Medibot-using-RAG/
│
├── data/
│   └── Medical PDFs
│
├── vectorstore/
│   └── db_faiss/
│
├── create_memory_for_llm.py
├── connect_memory_with_llm.py
├── medibot.py
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/Sahil489s/Medibot-using-RAG.git
cd Medibot-using-RAG
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the root directory:

```env
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
```

---

## 📚 Build the Vector Database

Before running the chatbot, create the FAISS vector store:

```bash
python create_memory_for_llm.py
```

This process:

* Loads medical PDFs
* Splits documents into chunks
* Creates embeddings
* Stores vectors in FAISS
* Saves the index locally

---

## 💬 Run the Application

### Streamlit Web App

```bash
streamlit run medibot.py
```

Open:

```text
http://localhost:8501
```

---

### CLI Version

```bash
python connect_memory_with_llm.py
```

---

## 🎯 Example Questions

Try asking:

* What are the symptoms of diabetes?
* What causes hypertension?
* Explain asthma.
* What is insulin therapy?
* How is pneumonia treated?
* What are the risk factors for heart disease?

---

## 📸 Demo

### Chat Interface

Add your screenshots here:

```md
![Chat UI](assets/chat_ui.png)
```

### Source Document Retrieval

```md
![Source Documents](assets/source_documents.png)
```

---

## 🔍 Sample Output

### User Question

```text
What are the symptoms of diabetes?
```

### Chatbot Response

```text
Common symptoms of diabetes include:

• Frequent urination
• Excessive thirst
• Increased hunger
• Fatigue
• Blurred vision
• Unexplained weight loss

Source:
Gale Encyclopedia of Medicine
```

---

## 🧪 Quick Verification

Verify the vector database:

```bash
python -c "
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

emb = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.load_local('vectorstore/db_faiss', emb, allow_dangerous_deserialization=True)

print(db.similarity_search('What is diabetes?', k=2))
"
```

---

## 🛠 Future Improvements

* 🎤 Voice-enabled chatbot
* 🧾 Chat history support
* 📊 RAG evaluation using RAGAS
* 🌐 Multi-document ingestion
* 🔄 Streaming responses
* 🔒 User authentication
* 🐳 Docker deployment
* ☁️ Cloud deployment

---

## 📈 Resume Project Description

Developed an AI-powered Medical Chatbot using Retrieval-Augmented Generation (RAG), LangChain, FAISS, HuggingFace Embeddings, and Groq-hosted LLMs. Implemented semantic search over medical PDFs to generate context-aware, source-grounded responses while minimizing hallucinations. Built an interactive Streamlit web application with document retrieval and source traceability.

---

## ⚠️ Disclaimer


It does not provide medical advice, diagnosis, or treatment recommendations. Always consult qualified healthcare professionals for medical decisions.

---

## 👨‍💻 Author

### Sahil Sharma

* Data Science & AI/ML Enthusiast
* Generative AI Developer
* RAG & LLM Applications Builder

GitHub: https://github.com/Sahil489s

---

## ⭐ Support

If you found this project useful:

* ⭐ Star the repository
* 🍴 Fork the repository
* 🛠️ Contribute to the project

---

<div align="center">

### 🚀 Built with LangChain, FAISS, HuggingFace & Groq

</div>
