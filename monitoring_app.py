import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px
from datetime import datetime
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# This is the correct way to get the secrets from the Streamlit Cloud environment
DRIVE_FOLDER_ID = "12QDMyXUbPJRlMsii2IcpdHgyGKpyIlcZ" # Replace with your actual Google Drive folder ID

def authenticate_gdrive():
    """Authenticates with Google Drive using secrets.toml."""
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        gauth.LoadClientSecretsFromDict(st.secrets["google_drive"]["client_secrets"])
        gauth.Authorize()
        gauth.SaveCredentialsFile("mycreds.txt")
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    return GoogleDrive(gauth)

def download_db_from_gdrive():
    """Downloads monitoring.db from Google Drive."""
    try:
        drive = authenticate_gdrive()
        file_list = drive.ListFile({'q': f"'{DRIVE_FOLDER_ID}' in parents and trashed=false"}).GetList()

        for file in file_list:
            if file['title'] == 'monitoring.db':
                gfile = drive.CreateFile({'id': file['id']})
                gfile.GetContentFile('monitoring.db')
                st.toast("Database downloaded successfully!", icon="⬇️")
                return
        st.warning("monitoring.db not found in cloud storage.")
    except Exception as e:
        st.error(f"Error downloading database from Google Drive: {e}")

# Call this function at the very beginning of your script,
# before you load any data from the database.
download_db_from_gdrive()

def get_monitoring_data():
    """Fetches all data from the local SQLite database."""
    try:
        conn = sqlite3.connect('monitoring.db')
        df = pd.read_sql_query("SELECT * FROM chat_logs", conn)
        conn.close()
        return df
    except sqlite3.OperationalError:
        return pd.DataFrame()

# Main function to display the dashboard
st.title("RAG Chatbot Monitoring Dashboard")

df = get_monitoring_data()

if not df.empty:
    st.subheader("Recent User Queries")
    st.dataframe(df.tail(10))

    # Basic Metrics
    st.subheader("Chatbot Metrics")
    total_interactions = len(df)
    unique_users = df['user_query'].nunique() # This is a simple proxy
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Interactions", value=total_interactions)
    with col2:
        st.metric(label="Unique Queries", value=unique_users)

    # ... (the rest of your dashboard code for visualizations) ...
else:
    st.warning("No monitoring data available yet. Run the main chatbot application to generate some logs.")
    
A video explaining how to integrate Python with Google Drive can be found here: [Upload to Google Drive with Python](https://www.youtube.com/watch?v=tamT_iGoZDQ).
http://googleusercontent.com/youtube_content/3
