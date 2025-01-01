from typing import Protocol, Dict, Any
from openai import OpenAI
import requests

class LLMProvider(Protocol):
    def get_completion(self, content: str, prompt: str) -> str:
        pass

class OpenAIProvider:
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def get_completion(self, content:str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt + "\n" + content}],
            temperature=0,
            max_tokens=10
        )
        return response.choices[0].message.content.strip().lower()

class OllamaProvider:
    def __init__(self, model: str = "llama3.1"):
        self.model = model
        self.base_url = "http://localhost:11434/api/generate"
    
    def get_completion(self, content: str, prompt: str) -> str:
        response = requests.post(
            self.base_url,
            json={"model": self.model, "prompt": prompt + "\n" + content, "stream": False}
        )
        response.raise_for_status()
        return response.json()["response"].strip().lower() 