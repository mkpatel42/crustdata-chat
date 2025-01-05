import re
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
import os
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import json
import logging
from langchain.schema import HumanMessage, AIMessage

with open('data.txt', 'r') as file:
    test_data = file.read()

# Page config
st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("Document Question & Answer System")

# API Key Management in Sidebar
with st.sidebar:
    st.header("API Configuration")
    if 'groq_api_key' not in st.session_state:
        groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
        if groq_api_key:
            st.session_state.groq_api_key = groq_api_key

# Initialize components only if API key is provided and chain not already created
if 'groq_api_key' in st.session_state and 'retrieval_chain' not in st.session_state:
    try:
        # Create document from static data
        doc = Document(page_content=test_data)

        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

        # Split document
        documents = text_splitter.split_documents([doc])

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Initialize LLM
        llm = ChatGroq(
            api_key=st.session_state.groq_api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0.7
        )

        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context:
        <context>
        {context}
        </context>

        Question: {input}
        """)

        # Create vector store
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )

        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Create retrieval chain
        st.session_state.retrieval_chain = create_retrieval_chain(retriever, document_chain)

    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")

# User Interface
# User Interface
if 'groq_api_key' not in st.session_state:
    st.warning("Please enter your Groq API key in the sidebar to continue.")
elif 'retrieval_chain' not in st.session_state:
    st.warning("System initialization in progress...")
else:
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Create a form for the question input
    with st.form(key='question_form'):
        user_question = st.text_input("Enter your question about Crustdata:")
        submit_button = st.form_submit_button(label='Ask')

    if submit_button and user_question:
        with st.spinner('Processing your question...'):
            try:
                response = st.session_state.retrieval_chain.invoke({
                    "input": user_question,
                    "chat_history": st.session_state.chat_history
                })
                
                # Store user question and AI response in chat history
                st.session_state.chat_history.insert(0, AIMessage(content=response['answer']))
                st.session_state.chat_history.insert(0, HumanMessage(content=user_question))

                # Limit chat history to prevent excessive memory usage
                st.session_state.chat_history = st.session_state.chat_history[:10]

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Display chat history
    for i in range(0, len(st.session_state.chat_history), 2):
        if i+1 < len(st.session_state.chat_history):
            human_message = st.session_state.chat_history[i]
            ai_message = st.session_state.chat_history[i+1]
            
            st.chat_message("human").write(human_message.content)
            st.chat_message("ai").write(ai_message.content)