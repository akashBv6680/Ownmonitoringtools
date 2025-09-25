import streamlit as st
import pysqlite3 as sqlite3
import os
import shutil
import tempfile
import chromadb
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from datetime import datetime
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# This is the correct way to get the secrets from the Streamlit Cloud environment
DRIVE_FOLDER_ID = "12QDMyXUbPJRlMsii2IcpdHgyGKpyIlcZ" # Replace with your actual Google Drive folder ID



def authenticate_gdrive():
    """Authenticates with Google Drive using secrets.toml."""
    gauth = GoogleAuth()
    
    # Create the credentials JSON object from the individual secrets
    client_secrets = {
        "installed": {
            "client_id": st.secrets["google_drive"]["client_id"],
            "project_id": st.secrets["google_drive"]["project_id"],
            "auth_uri": st.secrets["google_drive"]["auth_uri"],
            "token_uri": st.secrets["google_drive"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["google_drive"]["auth_provider_x509_cert_url"],
            "client_secret": st.secrets["google_drive"]["client_secret"],
            "redirect_uris": [st.secrets["google_drive"]["redirect_uris"]]
        }
    }

    # Set the client configuration in the GoogleAuth settings
    gauth.settings["client_config"] = client_secrets["installed"]
    gauth.LoadCredentialsFile("mycreds.txt")

    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
        gauth.SaveCredentialsFile("mycreds.txt")
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
        
    return GoogleDrive(gauth)

def upload_db_to_gdrive():
    """Uploads the local monitoring.db to Google Drive."""
    try:
        drive = authenticate_gdrive()
        file_list = drive.ListFile({'q': f"'{DRIVE_FOLDER_ID}' in parents and trashed=false"}).GetList()
        file_id = None
        for file in file_list:
            if file['title'] == 'monitoring.db':
                file_id = file['id']
                break

        if file_id:
            gfile = drive.CreateFile({'id': file_id})
        else:
            gfile = drive.CreateFile({'title': 'monitoring.db', 'parents': [{'id': DRIVE_FOLDER_ID}]})
        
        gfile.SetContentFile('monitoring.db')
        gfile.Upload()
        st.toast("Database uploaded successfully!", icon="☁️")

    except Exception as e:
        st.error(f"Error uploading database to Google Drive: {e}")

# Function to initialize the database
def init_db():
    conn = sqlite3.connect('monitoring.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_query TEXT,
            model_response TEXT,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    conn.close()

# Function to log chat interaction
def log_interaction(query, response):
    try:
        conn = sqlite3.connect('monitoring.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO chat_logs (user_query, model_response, timestamp) VALUES (?, ?, ?)",
                       (query, response, datetime.now()))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error logging interaction: {e}")
        
def load_and_split_pdf(pdf_path):
    """Loads a PDF and splits it into text chunks."""
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    return docs

def setup_chroma_db(docs):
    """Initializes and returns a ChromaDB vector store."""
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore

def setup_qa_chain(vectorstore):
    """Initializes and returns a RetrievalQA chain."""
    llm = Ollama(model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )
    return qa_chain

def handle_user_input(qa_chain):
    """Handles the user's chat input and provides a response."""
    user_query = st.text_input("Ask a question about the PDF:", key="user_query")
    if user_query:
        try:
            with st.spinner("Getting response..."):
                response = qa_chain.run(user_query)
                st.session_state.history.append({"query": user_query, "response": response})
                log_interaction(user_query, response)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Main application logic
if "history" not in st.session_state:
    st.session_state.history = []

st.title("RAG Chatbot with Ollama")

# Initialize database on app start
init_db()

# This is where you would load your PDF
# Make sure your PDF is available in the app's directory
pdf_path = "your_document.pdf"

if os.path.exists(pdf_path):
    docs = load_and_split_pdf(pdf_path)
    vectorstore = setup_chroma_db(docs)
    qa_chain = setup_qa_chain(vectorstore)
    handle_user_input(qa_chain)
else:
    st.warning(f"PDF file '{pdf_path}' not found. Please upload it to your app's directory.")
    
# Display chat history
if st.session_state.history:
    st.subheader("Chat History")
    for chat in reversed(st.session_state.history):
        st.write(f"**You:** {chat['query']}")
        st.write(f"**Bot:** {chat['response']}")
        
# Call the upload function right after closing the connection in your log_interaction function
# to ensure it runs after a user message.
