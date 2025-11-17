import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
import tempfile
import os

# ⚠️ Replace this safely with st.secrets["GROQ_API_KEY"]
GROQ_API_KEY=""


# ✅ Input guardrail – check if user query relates to document
def is_query_relevant(query: str, top_docs):
    if not top_docs:
        return False
    combined_context = " ".join([d.page_content for d in top_docs]).lower()
    query = query.lower()

    # Find matching words
    common_words = [w for w in query.split() if w in combined_context]
    return len(common_words) > 2  # Tune threshold as needed

# App header
st.header("📚 Chat with your PDF ")

# Sidebar upload
with st.sidebar:
    st.title("Upload your PDF")
    file = st.file_uploader("Upload a PDF to chat with", type="pdf")

# If file uploaded
if file is not None:
    # Extract text from PDF
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings (using MiniLM — fast + local)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Use a temporary directory for Chroma
    persist_dir = tempfile.mkdtemp()

    # Create or load Chroma vector store
    vector_store = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=persist_dir)

    # User query
    user_question = st.text_input("💬 Ask something about your PDF:")

    if user_question:
        # Retrieve similar chunks
        relevant_docs = vector_store.similarity_search(user_question, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        if not is_query_relevant(user_question, relevant_docs):
            st.warning(
                "⚠️ Your question doesn’t seem related to the uploaded PDF. Please ask something from the document."
            )
            st.stop()

        # Build prompt
        prompt = f"""
        You are a helpful assistant.
        Use only the context below to answer.

        Context:
        {context}

        Question: {user_question}
        """

        # Create Groq LLM
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )

        # Generate response
        response = llm.invoke(prompt)

        # Display
        st.subheader("🧠 Answer:")
        st.write(response.content)

        # Optional: debug context
        with st.expander("🪄 View retrieved context"):
            st.write(context)
