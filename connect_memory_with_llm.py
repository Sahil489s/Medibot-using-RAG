import os

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
load_dotenv()

# Step 1: Setup Groq LLM
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.1-8b-instant"  # Change to any supported Groq model

llm = ChatGroq(
    model=GROQ_MODEL_NAME,
    temperature=0.5,
    max_tokens=512,
    api_key=GROQ_API_KEY,
)

# Step 2: Load FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 3: Create retriever
retriever = db.as_retriever(search_kwargs={'k': 3})

# Step 4: Define prompt (replaces hub.pull("langchain-ai/retrieval-qa-chat"))
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful medical assistant. Use the following retrieved context to answer "
     "the user's question accurately. If you don't know the answer, say so clearly.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

# Step 5: Helper to format retrieved docs into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 6: Build LCEL RAG chain
#   - retriever fetches relevant docs, format_docs joins them into a string -> "context"
#   - RunnablePassthrough passes the user question as-is              -> "question"
#   - prompt fills both into the template
#   - llm generates the answer
#   - StrOutputParser extracts the plain text response
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Step 7: Also keep retriever separately so we can show source docs
user_query = input("Write Query Here: ")

# Get answer
response = rag_chain.invoke(user_query)
print("RESULT: ", response)

# Get source documents separately for display
source_docs = retriever.invoke(user_query)
print("\nSOURCE DOCUMENTS:")
for doc in source_docs:
    print(f"- {doc.metadata} -> {doc.page_content[:200]}...")
