"""
Build out the email dataset by connecting to Fastmail, pulling down sample emails,
and asking the user to label each one. This becomes the inputs (emails) and
expected outputs (labels) for evals and/or fine-tuning.
"""
import sqlite3
from typing import Dict, Any
import os, random

from fastmail_watcher import FastmailWatcher

def setup_database() -> sqlite3.Connection:
    """Create SQLite database and table if they don't exist."""
    conn = sqlite3.connect('email_dataset.sqlite')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS labeled_emails (
        email_id TEXT PRIMARY KEY,
        sender_name TEXT,
        sender_email TEXT,
        subject TEXT,
        body TEXT,
        label TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    return conn

def get_user_label() -> str:
    """Prompt user for email label until valid input received."""
    valid_labels = ['inbox', 'fyi', 'junk']
    while True:
        label = input(f"\nEnter label ({'/'.join(valid_labels)} or skip): ").lower().strip()
        if label == 'skip':
            return None
        if label in valid_labels:
            return label
        print(f"Invalid label. Please choose from: {', '.join(valid_labels)}")

def process_email(email: Dict[str, Any], conn: sqlite3.Connection) -> None:
    """Save labeled email to database."""
    cursor = conn.cursor()
    
    # Check if email already labeled
    cursor.execute('SELECT email_id FROM labeled_emails WHERE email_id = ?', (email['id'],))
    if cursor.fetchone():
        print("Email already labeled, skipping...")
        return

    sender = email['from'][0]['email']
    if sender == 'a@adamwiggins.com':
        return

    # Display email preview
    print(f"\nFrom: {email['from'][0]['name']} <{email['from'][0]['email']}>")
    print(f"Subject: {email['subject']}")
    print("\nPreview:")
    print(email['body'][:500] + "...\n")
    
    # Get label from user
    label = get_user_label()
    if label is None:
        return
    
    # Save to database
    cursor.execute('''
    INSERT INTO labeled_emails 
    (email_id, sender_name, sender_email, subject, body, label)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        email['id'],
        email['from'][0]['name'],
        email['from'][0]['email'],
        email['subject'],
        email['body'],
        label
    ))
    conn.commit()

def build_dataset():
    # Storing the resulting dataset in SQLite
    conn = setup_database()
    
    # Fetch source emails from Fastmail
    api_token = os.getenv("FASTMAIL_API_TOKEN")
    if not api_token:
        raise ValueError("Please set FASTMAIL_API_TOKEN environment variable")
    
    watcher = FastmailWatcher(api_token)
    
    while True:
        print("... Fetching some emails from Fastmail ...")
        emails = watcher.get_recent_emails(limit=5, offset=random.randint(0, 10000))
        
        # Process each email
        for email in emails:
            try:
                process_email(email, conn)
            except KeyboardInterrupt:
                print("\nStopping dataset collection...")
                break
            except Exception as e:
                print(f"Error processing email: {e}")
                continue
    
    conn.close()

if __name__ == "__main__":
    build_dataset()
