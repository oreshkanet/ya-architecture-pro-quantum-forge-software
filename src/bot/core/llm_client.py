"""
Ollama client for RAG Bot (Wiki Universe)
"""
import requests
from typing import Optional
from .config import OLLAMA_API_URL, OLLAMA_MODEL, OLLAMA_MAX_TOKENS, OLLAMA_TEMPERATURE


class OllamaClient:
    def __init__(self):
        self.url = OLLAMA_API_URL
        self.model = OLLAMA_MODEL

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": OLLAMA_TEMPERATURE,
                "num_predict": OLLAMA_MAX_TOKENS,
            }
        }
        try:
            response = requests.post(f"{self.url}/api/generate", json=payload, timeout=600)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            return f"❌ Ошибка генерации: {e}"