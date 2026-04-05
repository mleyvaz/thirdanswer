"""Base provider interface."""

from abc import ABC, abstractmethod
from typing import Optional


class Provider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, system: str, user: str, temperature: float = 0.3,
                 max_tokens: int = 2000) -> str:
        """Send a completion request and return raw text."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
