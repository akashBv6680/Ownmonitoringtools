import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import time

# --- Database and Data Loading Functions ---
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

def get_monitoring_data():
    """Fetches all data from the SQLite database."""
    try:
        conn = sqlite3.connect('monitoring.db')
        df = pd.read_sql_query("SELECT * FROM chat_logs", conn)
        conn.close()
        return df
    except sqlite3.OperationalError:
        # This will catch the error if the database file is not yet created
        # and return an empty dataframe
        return pd.DataFrame()

# --- Streamlit UI ---
st.set_page_config(layout="wide")

st.title("RAG Chatbot Monitoring Dashboard")
st.markdown("This dashboard provides a live view of your chatbot's performance.")

# Button to refresh data
if st.button("Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# Load data with caching
@st.cache_data(ttl=60)
def load_data():
    return get_monitoring_data()

monitoring_df = load_data()

if not monitoring_df.empty:
    # Convert timestamp to datetime for proper sorting and plotting
    monitoring_df['timestamp'] = pd.to_datetime(monitoring_df['timestamp'])

    # Metrics at the top for a quick overview
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Latency", f"{monitoring_df['latency'].mean():.2f}s")
    col2.metric("Total Queries", len(monitoring_df))
    col3.metric("Total Errors", len(monitoring_df[monitoring_df['error_status'] == 'Yes']))

    st.markdown("---")

    st.subheader("Latency Over Time")
    st.line_chart(monitoring_df.set_index('timestamp')['latency'])

    st.subheader("Tool Usage")
    tool_counts = monitoring_df['tool_used'].value_counts()
    st.bar_chart(tool_counts)

    st.subheader("Detailed Logs")
    st.dataframe(monitoring_df.sort_values(by='timestamp', ascending=False), use_container_width=True)

else:
    st.info("No monitoring data available yet. Run the main chatbot application to generate some logs.")
