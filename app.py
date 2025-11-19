import streamlit as st
import json
import os
import tempfile

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# -------------------------
# Load API Key
# -------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------------
# JSON File Path
# -------------------------
JSON_PATH = "knowledge.json"


# -------------------------
# Load JSON Knowledge Base
# -------------------------
def load_json(path=JSON_PATH):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for entry in data:
        text = entry.get("content", "")
        metadata = {
            "id": entry.get("id"),
            "title": entry.get("title")
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


# -------------------------
# Input Relevance Check
# -------------------------
def is_query_relevant(query, docs):
    if not docs:
        return False

    combined = " ".join([d.page_content for d in docs]).lower()
    query = query.lower()

    common_words = [w for w in query.split() if w in combined]
    return len(common_words) > 1


# -------------------------
# Build Vector DB
# -------------------------
docs = load_json()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

all_chunks = []
for doc in docs:
    chunks = splitter.split_text(doc.page_content)
    for chunk in chunks:
        all_chunks.append(Document(page_content=chunk, metadata=doc.metadata))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

persist_dir = tempfile.mkdtemp()

vector_store = Chroma.from_documents(
    all_chunks,
    embedding=embeddings,
    persist_directory=persist_dir
)

# -------------------------
# UI Setup
# -------------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 JSON RAG Chatbot")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []


# -------------------------
# Chat Input
# -------------------------
def generate_answer(user_input):
    retrieved = vector_store.similarity_search(user_input, k=3)

    if not is_query_relevant(user_input, retrieved):
        return "⚠️ I can only answer questions related to the JSON knowledge base."

    context = "\n\n".join(
        [f"[{d.metadata}] {d.page_content}" for d in retrieved]
    )

    prompt = f"""
    You are a knowledgeable AI assistant.
    You must answer ONLY using the following JSON context:

    Context:
    {context}

    Question: {user_input}

    If answer is not in the context, reply:
    "I can only answer questions related to the JSON knowledge base."
    """

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    response = llm.invoke(prompt)
    return response.content


# -------------------------
# Display Chat History UI (ChatGPT style)
# -------------------------
for chat in st.session_state.history:
    if chat["role"] == "user":
        st.chat_message("user").markdown(chat["content"])
    else:
        st.chat_message("assistant").markdown(chat["content"])


# -------------------------
# Chat Input Box
# -------------------------
user_input = st.chat_input("Ask your question...")

if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)

    # Generate answer
    answer = generate_answer(user_input)

    # Append history
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": answer})

    # Display assistant message
    st.chat_message("assistant").markdown(answer)
