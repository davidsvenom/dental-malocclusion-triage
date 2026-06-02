import sqlite3
import os
from datetime import datetime

# Define database path
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, 'triage_logs.db')

def init_db():
    """Initializes the database and creates the table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS triage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_filename TEXT,
            num_detections INTEGER,
            confidence_score REAL,
            severity_level TEXT,
            triage_recommendation TEXT,
            model_version TEXT,
            flag_for_review TEXT,
            detected_classes TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def insert_log(image_filename, num_detections, confidence_score, severity_level, triage_recommendation, model_version, flag_for_review, detected_classes):
    """Inserts a new log entry into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO triage_logs (
            timestamp, 
            image_filename, 
            num_detections, 
            confidence_score, 
            severity_level, 
            triage_recommendation, 
            model_version, 
            flag_for_review,
            detected_classes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (current_time, image_filename, num_detections, confidence_score, severity_level, triage_recommendation, model_version, str(flag_for_review), detected_classes))
    
    conn.commit()
    conn.close()

def get_all_logs():
    """Retrieves all logs from the database, ordered by newest first."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name: row['column_name']
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM triage_logs ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    
    conn.close()
    return [dict(row) for row in rows]

# Initialize the DB when the module is imported
init_db()
