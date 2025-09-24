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
