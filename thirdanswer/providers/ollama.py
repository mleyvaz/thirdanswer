"""Ollama provider — local, free, private."""

import json
import urllib.request
from .base import Provider


class OllamaProvider(Provider):
    """
    Ollama local inference. No API key needed.
    Install: https://ollama.com/download
    """

    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434"):
        self._model = model
        self._host = host

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    def complete(self, system: str, user: str, temperature: float = 0.3,
                 max_tokens: int = 2000) -> str:
        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode()

        req = urllib.request.Request(
            f"{self._host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["message"]["content"].strip()
