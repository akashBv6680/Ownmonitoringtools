import streamlit as st
import os
import sys
import tempfile
import uuid
import json
import requests
import time
from datetime import datetime
import re
from typing import List
import fitz # Import the PyMuPDF library for PDF processing

# --- Set Page Config (Must be the very first Streamlit command) ---
st.set_page_config(layout="wide")

# This block MUST be at the very top to fix the sqlite3 version issue.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    st.error("pysqlite3 is not installed. Please add 'pysqlite3-binary' to your requirements.txt.")
    st.stop()

# Set LangSmith environment variables from Streamlit secrets
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

# Now import other libraries
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain import hub
from langchain_community.llms import Together
from sentence_transformers import SentenceTransformer
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup


# --- Constants and Configuration ---
COLLECTION_NAME = "agentic_rag_documents"
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"


# --- Centralized Session State Initialization ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.chat_history[st.session_state.current_chat_id] = {
        'messages': st.session_state.messages,
        'title': "New Chat",
        'date': datetime.now()
    }
    
@st.cache_resource
def initialize_dependencies():
    """
    Initializes and returns the ChromaDB client and SentenceTransformer model.
    Using @st.cache_resource ensures this runs only once.
    """
    try:
        db_path = tempfile.mkdtemp()
        db_client = chromadb.PersistentClient(path=db_path)
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        return db_client, model
    except Exception as e:
        st.error(f"An error occurred during dependency initialization: {e}.")
        st.stop()

if 'db_client' not in st.session_state or 'model' not in st.session_state:
    st.session_state.db_client, st.session_state.model = initialize_dependencies()


def get_collection():
    """Retrieves or creates the ChromaDB collection."""
    return st.session_state.db_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

@tool
def retrieve_documents(query: str) -> str:
    """
    USE THIS TOOL ONLY when the user asks a question about the uploaded documents, GitHub URL content, or website content.
    This tool retrieves specific information from the provided knowledge base.
    """
    try:
        collection = get_collection()
        model = st.session_state.model
        query_embedding = model.encode(query).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=5
        )
        # Check if results are empty and provide a clear message
        if not results['documents'][0]:
            return "No relevant documents were found in the knowledge base. Please use another tool or ask a different question."
        return "\n".join(results['documents'][0])
    except Exception as e:
        return f"An error occurred during document retrieval: {e}"

@tool
def calculator(expression: str) -> str:
    """
    Calculates the result of a mathematical expression string.
    This tool is useful for simple calculations (e.g., "2 * 3 + 5").
    """
    try:
        # Use eval() with caution as it can be a security risk in production environments
        return str(eval(expression))
    except Exception as e:
        return f"Error: Could not evaluate expression. {e}"

@tool
def duckduckgo_search(query: str) -> str:
    """
    USE THIS TOOL ONLY for questions about current events, general knowledge, or information not found in the uploaded documents.
    Do NOT use this tool to answer questions about the provided knowledge base (documents, URLs).
    """
    search = DuckDuckGoSearchRun()
    return search.run(query)

def create_agent():
    """Creates and returns a LangChain agent executor."""
    
    # Custom prompt to give the agent a better sense of when to use each tool
    custom_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to specialized tools. You should use the provided tools to answer the user's questions. Follow the ReAct framework to reason and act. Your primary goal is to provide accurate answers. If a question can be answered by the documents, prioritize the 'retrieve_documents' tool. If it's a general knowledge question or about current events, use 'duckduckgo_search'. For math problems, use the 'calculator'. Your final answer should be concise and direct.\n\nHere are the tools you have access to:\n{tools}\n\nUse the following format to interact with the tools:\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n...(this Thought/Action/Observation loop can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\n{agent_scratchpad}"),
        ("human", "{input}"),
    ])
    
    tools = [
        retrieve_documents,
        calculator,
        duckduckgo_search
    ]
    
    together_llm = Together(
        together_api_key=TOGETHER_API_KEY,
        model="mistralai/Mistral-7B-Instruct-v0.2"
    )
    
    agent = create_react_agent(together_llm, tools, custom_prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def clear_chroma_data():
    """Clears all data from the ChromaDB collection."""
    try:
        if COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
            st.session_state.db_client.delete_collection(name=COLLECTION_NAME)
    except Exception as e:
        st.error(f"Error clearing collection: {e}")

def split_documents(text_data, chunk_size=500, chunk_overlap=100) -> List[str]:
    """Splits a single string of text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text_data)

def process_and_store_documents(documents: List[str]):
    """
    Processes a list of text documents, generates embeddings, and
    stores them in ChromaDB.
    """
    collection = get_collection()
    model = st.session_state.model

    embeddings = model.encode(documents).tolist()
    document_ids = [str(uuid.uuid4()) for _ in documents]
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=document_ids
    )
    st.toast("Documents processed and stored successfully!", icon="✅")

def is_valid_github_raw_url(url: str) -> bool:
    """Checks if a URL is a valid GitHub raw file URL."""
    pattern = r"https://raw\.githubusercontent\.com/[\w-]+/[\w-]+/[^/]+/[\w./-]+\.(txt|md)"
    return re.match(pattern, url) is not None

def display_chat_messages():
    """Displays all chat messages in the Streamlit app."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input():
    """Handles new user input, runs the RAG pipeline, and updates chat history."""
    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                agent_executor = create_agent()
                try:
                    response = agent_executor.invoke({"input": prompt, "chat_history": st.session_state.messages})
                    final_response = response.get('output', 'An error occurred.')
                except Exception as e:
                    final_response = f"An error occurred: {e}"
                st.markdown(final_response)
                
        # Fix the AttributeError by correcting the typo
        st.session_state.messages.append({"role": "assistant", "content": final_response})

# --- Main UI ---
st.title("Agentic RAG Chat Flow")
st.markdown("---")

# Document upload/processing section
with st.container():
    st.subheader("Add Context Documents")
    uploaded_files = st.file_uploader("Upload text (.txt) or PDF files (.pdf)", type=["txt", "pdf"], accept_multiple_files=True)
    github_url = st.text_input("Enter a GitHub raw `.txt` or `.md` URL:")
    website_url = st.text_input("Enter a Website URL:")
    
    if uploaded_files:
        if st.button("Process Files"):
            with st.spinner("Processing files..."):
                all_text = ""
                for uploaded_file in uploaded_files:
                    if uploaded_file.type == "text/plain":
                        file_contents = uploaded_file.read().decode("utf-8")
                        all_text += file_contents + "\n" # Add a newline to separate content from different files
                    elif uploaded_file.type == "application/pdf":
                        try:
                            # Use PyMuPDF (fitz) to open the PDF in memory
                            doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
                            for page in doc:
                                all_text += page.get_text()
                            doc.close()
                        except Exception as e:
                            st.error(f"Error processing PDF file: {uploaded_file.name}. Error: {e}")
                            
                if all_text:
                    documents = split_documents(all_text)
                    process_and_store_documents(documents)
                    st.success("All files processed and stored successfully! You can now ask questions about their content.")
                else:
                    st.warning("No text could be extracted from the uploaded files.")

    if github_url and is_valid_github_raw_url(github_url):
        if st.button("Process GitHub URL"):
            with st.spinner("Fetching and processing file from URL..."):
                try:
                    response = requests.get(github_url)
                    response.raise_for_status()
                    file_contents = response.text
                    documents = split_documents(file_contents)
                    process_and_store_documents(documents)
                    st.success("File from URL processed! You can now chat about its contents.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching URL: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
    
    if website_url:
        if st.button("Process Website URL"):
            with st.spinner("Fetching and processing website content..."):
                try:
                    # Use WebBaseLoader to load and parse the website content
                    loader = WebBaseLoader(website_url)
                    docs = loader.load()
                    
                    if not docs:
                        st.warning("No content found at the provided URL.")
                    else:
                        full_text = " ".join([d.page_content for d in docs])
                        documents = split_documents(full_text)
                        process_and_store_documents(documents)
                        st.success("Website content processed! You can now chat about its content.")

                except Exception as e:
                    st.error(f"Error processing website URL: {e}")

# Sidebar
with st.sidebar:
    st.header("Agentic RAG Chat Flow")
    if st.button("New Chat"):
        st.session_state.messages = []
        clear_chroma_data()
        st.session_state.chat_history = {}
        st.session_state.current_chat_id = None
        st.experimental_rerun()

    st.subheader("Chat History")
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        sorted_chat_ids = sorted(
            st.session_state.chat_history.keys(),
            key=lambda x: st.session_state.chat_history[x]['date'],
            reverse=True
        )
        for chat_id in sorted_chat_ids:
            chat_title = st.session_state.chat_history[chat_id]['title']
            date_str = st.session_state.chat_history[chat_id]['date'].strftime("%b %d, %I:%M %p")
            if st.button(f"**{chat_title}** - {date_str}", key=chat_id):
                st.session_state.current_chat_id = chat_id
                st.session_state.messages = st.session_state.chat_history[chat_id]['messages']
                st.experimental_rerun()

display_chat_messages()
handle_user_input()

# --- In your streamlit_app.py file ---

# ... (other code) ...

def handle_user_input():
    """Handles new user input, runs the RAG pipeline, and logs data to DB."""
    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        start_time = time.time()
        agent_executor = create_agent()
        final_response = "An error occurred."
        tool_used = "N/A"
        retrieved_docs_count = 0
        error_status = "No"

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = agent_executor.invoke({"input": prompt, "chat_history": st.session_state.messages})
                    final_response = response.get('output', 'No response.')

                    # Infer tool used (simplified)
                    if "using `retrieve_documents`" in final_response:
                        tool_used = "RAG"
                    elif "using `duckduckgo_search`" in final_response:
                        tool_used = "DuckDuckGo"
                    elif "using `calculator`" in final_response:
                        tool_used = "Calculator"

                except Exception as e:
                    final_response = f"An error occurred: {e}"
                    error_status = "Yes"

            latency = time.time() - start_time
            st.markdown(final_response)

        st.session_state.messages.append({"role": "assistant", "content": final_response})

        # --- Log to Database ---
        conn = sqlite3.connect('monitoring.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_logs (timestamp, query, latency, tool_used, retrieved_docs_count, error_status) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), prompt, latency, tool_used, 0, error_status) # retrieved_docs_count is hardcoded here for simplicity
        )
        conn.commit()
        conn.close()

# ... (rest of your main app code) ...

import sqlite3
import pandas as pd
# ... other imports ...

def init_db():
    """Initializes a SQLite database and the monitoring table."""
    conn = sqlite3.connect('monitoring.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            latency REAL,
            tool_used TEXT,
            retrieved_docs_count INTEGER,
            error_status TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Call this function once at the beginning of your app
init_db()
