import requests
import json

from bs4 import BeautifulSoup
import re

import time
from typing import Optional, Dict, Any
import os

from openai import OpenAI

from classify_email import classify_email
from providers import OpenAIProvider, OllamaProvider

def clean_html_email_with_markdown(html_content):
    """
    Convert HTML email content to markdown while preserving basic formatting.
    
    Args:
        html_content (str): HTML content from email
        
    Returns:
        str: Markdown version of the content
    """
    if not html_content or not isinstance(html_content, str):
        return ""
    
    # Initialize BeautifulSoup
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
    except Exception:
        soup = BeautifulSoup(html_content, 'html5lib')
    
    # Remove unwanted elements
    for element in soup(['script', 'style', 'head']):
        element.decompose()
    
    # Convert common HTML elements to markdown
    for tag in soup.find_all(['strong', 'b']):
        tag.replace_with(f'**{tag.get_text()}**')
    
    for tag in soup.find_all(['em', 'i']):
        tag.replace_with(f'*{tag.get_text()}*')
    
    for tag in soup.find_all('a'):
        url = tag.get('href', '')
        text = tag.get_text()
        tag.replace_with(f'[{text}]({url})')
    
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        level = int(tag.name[1])
        text = tag.get_text()
        tag.replace_with(f'\n{"#" * level} {text}\n')
    
    # Convert lists
    for tag in soup.find_all(['ul', 'ol']):
        items = tag.find_all('li')
        for i, item in enumerate(items):
            prefix = '* ' if tag.name == 'ul' else f'{i+1}. '
            item.replace_with(f'\n{prefix}{item.get_text()}')
    
    # Get final text
    text = soup.get_text(separator=' ')
    
    # Clean up whitespace and special characters
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    
    return text

class FastmailWatcher:
    def __init__(self, api_token: str, provider_type: str = "openai", model: str = "gpt-4"):
        self.api_token = api_token
        self.base_url = "https://api.fastmail.com/jmap/api/"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        self.account_id = self._get_account_id()

    def _get_account_id(self) -> str:
        session_response = requests.get(
            "https://api.fastmail.com/jmap/session",
            headers=self.headers
        )
        session_response.raise_for_status()
        return session_response.json()["primaryAccounts"]["urn:ietf:params:jmap:mail"]

    def _get_email_details(self, email_id: str) -> Dict[str, Any]:
        request_data = {
            "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
            "methodCalls": [[
                "Email/get",
                {
                    "accountId": self.account_id,
                    "ids": [email_id],
                    "properties": ["subject", "from", "to", "bodyValues", "textBody", "htmlBody"],
                    "fetchTextBodyValues": True,
                    "fetchHTMLBodyValues": True
                },
                "a"
            ]]
        }
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=request_data
        )
        response.raise_for_status()
      
        result = response.json()["methodResponses"][0][1]
        email = result["list"][0]
        
        # Try different possible body part IDs
        body_part_ids = ['1', '1.1']  # Add more if needed
        body = None
        
        for part_id in body_part_ids:
            if part_id in email["bodyValues"]:
                body = email["bodyValues"][part_id]['value']
                break
        
        if body is None:
            # If no recognized part is found, log available parts and use the first one
            available_parts = list(email["bodyValues"].keys())
            print(f"Warning: Expected body parts not found. Available parts: {available_parts}")
            if available_parts:
                body = email["bodyValues"][available_parts[0]]['value']
            else:
                body = ""
        
        email['body'] = clean_html_email_with_markdown(body)
        return email

    def check_new_emails(self) -> None:
        request_data = {
            "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
            "methodCalls": [[
                "Email/query",
                {
                    "accountId": self.account_id,
                    "sort": [{"property": "receivedAt", "isAscending": False}],
                    "limit": 10
                },
                "a"
            ]]
        }
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=request_data
        )
        response.raise_for_status()
        
        email_ids = response.json()["methodResponses"][0][1]["ids"]
        
        for email_id in reversed(email_ids):
            email = self._get_email_details(email_id)
            print(f"From: {email['from'][0]['name']} {email['from'][0]['email']}")
            print(f"Subject: {email['subject']}")
            print()
            print(email['body'][:1000])
            print('-' * 80)

            label = classify_email(self.provider, email['body'])
            print(f"Classifer tags this as: {label}")
            print('-' * 80)

    def watch(self, interval: int = 60) -> None:
        # Initialize the appropriate provider
        if provider_type == "openai":
            self.provider = OpenAIProvider(model=model)
        elif provider_type == "ollama":
            self.provider = OllamaProvider(model=model)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

        print("Checking fastmail for recent emails")
        self.check_new_emails()

    def get_recent_emails(self, limit: int = 10, offset: int = 0) -> list:
        """Fetch recent emails and return their details."""
        request_data = {
            "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
            "methodCalls": [[
                "Email/query",
                {
                    "accountId": self.account_id,
                    "sort": [{"property": "receivedAt", "isAscending": False}],
                    "limit": limit,
                    "position": offset
                },
                "a"
            ]]
        }
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=request_data
        )
        response.raise_for_status()
        
        email_ids = response.json()["methodResponses"][0][1]["ids"]
        emails = []
        
        for email_id in reversed(email_ids):
            email = self._get_email_details(email_id)
            email['id'] = email_id  # Add the ID to the email dict
            emails.append(email)
            
        return emails

if __name__ == "__main__":
    api_token = os.getenv("FASTMAIL_API_TOKEN")
    if not api_token:
        raise ValueError("Please set FASTMAIL_API_TOKEN environment variable")
    
    watcher = FastmailWatcher(api_token)

    provider_type = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4o")
    
    watcher.watch(provider_type=provider_type, model=model)
