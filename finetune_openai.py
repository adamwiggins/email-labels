import sqlite3
import json
from pathlib import Path

SYSTEM_PROMPT = """You are my executive assistant, and you are excellent at sorting through my emails and labeling them as Inbox, FYI, or Junk.

Inbox includes personal and professional correspondance with real humans that I know. It also may include automated emails from services I use when they require my action, for example login links. Also in inbox: investor updates, calendar invites. If they reference one of my projects such as The Browser Company, Muse, Ink & Switch, Heroku, or Local-First Conf then they usually go to the inbox.

FYI includes order receipts (for example, from Amazon) and newsletters I've subscribed to such as Money Stuff, Tangle, Benedict Evans, Hacker Newsletter, Elicit, Butter Docs, and Kevin Lynagh. Also included in FYI: security alerts, Patreon project updates, and Readwise highlights. All newsletters from buttondown.email are in FYI.

Junk is any sale or promotion (even from a service I've purchased from) and newsletters that I never subscribed to. All Substack newsletters go to junk (I read them in the app instead).

Response to the email below the line with just the label, nothing else."""

def create_finetune_jsonl():
    """
    Convert labeled emails from SQLite database to JSONL format for GPT fine-tuning.
    Each example will be formatted as a system/user/assistant conversation.
    """
    # Connect to database
    db_path = 'datasets/for-finetuning.sqlite'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all labeled examples
    cursor.execute('SELECT body, sender_name, sender_email, subject, label FROM labeled_emails')
    examples = cursor.fetchall()
    
    # Create output file
    output_path = 'finetune_data.jsonl'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for body, sender_name, sender_email, subject, label in examples:
            # Format email content
            email_content = f"From: {sender_name} <{sender_email}>\nSubject: {subject}\n\n{body[:1000]}"
            
            # Create conversation format
            conversation = {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"---\n{email_content}"
                    },
                    {
                        "role": "assistant",
                        "content": label
                    }
                ]
            }
            
            # Write as JSONL
            f.write(json.dumps(conversation) + '\n')
    
    conn.close()
    print(f"Created fine-tuning dataset at {output_path}")

if __name__ == "__main__":
    create_finetune_jsonl() 