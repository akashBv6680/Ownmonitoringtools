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
from langchain_core.callbacks.manager import CallbackManager
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

    gauth.LoadClientSecretsFromDict(client_secrets)
    gauth.LoadCredentialsFile("mycreds.txt")
    
    if gauth.credentials is None:
        gauth.Authorize()
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

# ... (The rest of your code for Ollama, ChromaDB, etc.) ...
# Place these functions at the beginning of your script

# Inside your main application loop:
# In your handle_user_input() function, after saving the data to the database:
# ...
conn = sqlite3.connect('monitoring.db')
cursor = conn.cursor()
# ... (your INSERT statement here)
conn.commit()
conn.close()

# Call the upload function right after closing the connection
upload_db_to_gdrive()
