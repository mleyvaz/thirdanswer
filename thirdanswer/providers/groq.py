"""Groq provider — free, fast LLM inference."""

from .base import Provider


class GroqProvider(Provider):
    """
    Groq cloud inference. Free tier: ~30 req/min.
    Get API key at console.groq.com
    """

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Install groq: pip install groq")
        self._client = Groq(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return f"groq/{self._model}"

    def complete(self, system: str, user: str, temperature: float = 0.3,
                 max_tokens: int = 2000) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
