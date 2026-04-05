"""
Error taxonomy from Chapter 1 of The Third Answer.
Four types of AI error, ordered by detectability and danger.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AIError:
    """A detected AI error with type, severity, and explanation."""

    type: str
    severity: str
    detectability: str
    description: str
    explanation: Optional[str] = None

    @property
    def emoji(self) -> str:
        return {
            "fabrication": "\U0001f534",
            "distortion": "\U0001f7e0",
            "conflation": "\U0001f7e1",
            "confident_ignorance": "\u26ab",
        }.get(self.type, "\u2753")

    def __repr__(self) -> str:
        return f"{self.emoji} {self.type.upper()}: {self.explanation or self.description}"


ERROR_TYPES = {
    "fabrication": {
        "name": "Fabrication",
        "severity": "high",
        "detectability": "medium",
        "description": "The AI invents something that doesn't exist and presents it as fact.",
        "example": "Citing a legal case that was never filed.",
        "detection_hint": "Search for the specific claim in authoritative databases.",
    },
    "distortion": {
        "name": "Distortion",
        "severity": "high",
        "detectability": "low",
        "description": "The AI takes a real fact and warps it.",
        "example": "'Moderate evidence' becomes 'strong evidence'.",
        "detection_hint": "Read the ORIGINAL source, not just the AI summary.",
    },
    "conflation": {
        "name": "Conflation",
        "severity": "medium",
        "detectability": "medium",
        "description": "The AI merges two true things into one false thing.",
        "example": "A real author paired with a book they didn't write.",
        "detection_hint": "Verify each individual claim separately.",
    },
    "confident_ignorance": {
        "name": "Confident Ignorance",
        "severity": "very_high",
        "detectability": "very_low",
        "description": "The AI has no reliable information but produces a confident response anyway.",
        "example": "Inventing prevalence statistics for a rare disease in a remote province.",
        "detection_hint": "Ask: Does reliable data on this topic even EXIST?",
    },
}
