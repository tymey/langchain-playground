import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="Ask My Files - Gemini Hybrid", page_icon="📄")
st.title("📄 Ask My Files")
st.caption("Upload a PDF and chat with it using Gemini (AI Studio Key)")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "history" not in st.session_state:
    st.session_state.history = []

# Process uploaded PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing document..."):
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        st.session_state.vectorstore = vectorstore
        os.remove(tmp_path)
        st.success("Document processed! You can now chat below.")

# Show chat history
for turn in st.session_state.history:
    st.chat_message("user").write(turn["question"])
    st.chat_message("assistant").write(turn["answer"])

# Chat input
query = st.chat_input("Ask a question about your document")
if query and st.session_state.vectorstore:
    st.chat_message("user").write(query)

    with st.spinner("Thinking..."):
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        chat_history = ""
        for turn in st.session_state.history:
            chat_history += f"User: {turn['question']}\nAI: {turn['answer']}\n"

        prompt = f"Answer the question based only on the context below.\n\nContext:\n{context}\n\nChat History: \n{chat_history}\nQuestion: {query}\nAnswer:"

        model = genai.GenerativeModel("models/gemini-2.0-flash")
        response = model.generate_content(prompt)

        answer = response.text.strip()
        st.chat_message("assistant").write(answer)

        st.session_state.history.append({"question": query, "answer": answer})
elif query:
    st.warning("Please upload a PDF first.")