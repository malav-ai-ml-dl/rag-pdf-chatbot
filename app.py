import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

# Streamlit UI setup
st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content.")

# Load API keys from Streamlit Secrets
huggingface_api_key = st.secrets.get("HUGGINGFACE_API_KEY")
groq_api_key = st.secrets.get("GROQ_API_KEY")

# Validate API keys
if not huggingface_api_key:
    st.error("🚨 HUGGINGFACE_API_KEY is missing in Streamlit Secrets.")
if not groq_api_key:
    st.error("🚨 GROQ_API_KEY is missing in Streamlit Secrets.")

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    huggingface_api_key=huggingface_api_key
)

# Input the Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password", value=groq_api_key)

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for idx, uploaded_file in enumerate(uploaded_files):
            temp_pdf = f"./temp_{idx}.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)
            os.remove(temp_pdf)  # Clean up temporary files

        # Process documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Contextualization prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, formulate a standalone question without the chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # QA system prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for Q&A. Use retrieved context to answer concisely. If unsure, say so.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Chat history session management
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User interaction
        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            st.write("**Assistant:**", response['answer'])
            st.write("**Chat History:**", session_history.messages)
else:
    st.warning("🚨 Please enter the Groq API Key.")
