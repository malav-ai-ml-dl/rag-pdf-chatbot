import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

# Streamlit UI setup
st.title("Conversational RAG With PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content")

# User input for API keys
hf_api_key = st.text_input("Enter your HuggingFace API Key:", type="password")
groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
session_id = st.text_input("Session ID", value="default_session")

# Initialize session storage for chat history
if 'store' not in st.session_state:
    st.session_state.store = {}

# Proceed only if both API keys are provided
if hf_api_key and groq_api_key:
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", api_key=hf_api_key)

        # Initialize LLM
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

        # File uploader
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            documents = []
            for idx, uploaded_file in enumerate(uploaded_files):
                temp_pdf = f"./temp_{idx}.pdf"
                with open(temp_pdf, "wb") as file:
                    file.write(uploaded_file.getvalue())

                # Load PDF and extract text
                loader = PyPDFLoader(temp_pdf)
                docs = loader.load()
                documents.extend(docs)
                os.remove(temp_pdf)  # Clean up temporary files

            # Split documents into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)

            # Create a FAISS vectorstore
            vectorstore = FAISS.from_documents(splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()

            # Prompt for contextualizing queries
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given a chat history and the latest user question, formulate a standalone question without the chat history."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            # Prompt for answering questions
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an assistant for Q&A. Use retrieved context to answer concisely. If unsure, say so.\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            
            # Create the question-answer chain
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            # Function to manage chat history
            def get_session_history(session: str) -> BaseChatMessageHistory:
                if session not in st.session_state.store:
                    st.session_state.store[session] = ChatMessageHistory()
                return st.session_state.store[session]

            # Make the chain conversational
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            # User input section
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
            st.warning("Please upload at least one PDF to start chatting.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please enter both API keys to proceed.")
