"""
compare() — Compare two AI responses using T,I,F.

Usage:
    from thirdanswer import compare
    diff = compare(response_a, response_b, provider="groq", api_key="gsk_...")
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .compass import Compass, _classify_zone
from .analyzer import _parse_json, _get_provider


SYSTEM_PROMPT = """Compare two AI responses using the Third Answer framework.

For each response, assess T (Truth), I (Indeterminacy), F (Falsity) independently.

Then analyze:
1. Where they AGREE (both T high on same claim)
2. Where they CONTRADICT (one T high, other F high on same topic)
3. Where BOTH are uncertain (both I high)
4. Which is more epistemically honest (acknowledges limitations)

Respond in JSON:
{
  "response_a": {"T": 0.0, "I": 0.0, "F": 0.0, "zone": "..."},
  "response_b": {"T": 0.0, "I": 0.0, "F": 0.0, "zone": "..."},
  "agreement": 0.0,
  "agreements": ["point 1", "point 2"],
  "conflicts": ["conflict 1 with analysis"],
  "combined_zone": "consensus|ambiguity|contradiction|ignorance",
  "more_honest": "a|b|neither",
  "recommendation": "trust_a|trust_b|investigate|neither"
}"""


@dataclass
class ComparisonResult:
    """Result of comparing two AI responses."""
    response_a_compass: Compass
    response_b_compass: Compass
    agreement: float
    agreements: List[str]
    conflicts: List[str]
    combined_zone: str
    more_honest: str
    recommendation: str

    def __repr__(self) -> str:
        return (
            f"Comparison:\n"
            f"  A: {self.response_a_compass}\n"
            f"  B: {self.response_b_compass}\n"
            f"  Agreement: {self.agreement:.0%} | "
            f"Conflicts: {len(self.conflicts)} | "
            f"More honest: {self.more_honest} | "
            f"Rec: {self.recommendation}"
        )


def compare(
    response_a: str,
    response_b: str,
    provider: str = "groq",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> ComparisonResult:
    """
    Compare two AI responses and analyze where they agree, conflict, and which is more honest.

    Args:
        response_a: First AI response
        response_b: Second AI response
        provider: "groq" or "ollama"
        api_key: API key (not needed for ollama)

    Returns:
        ComparisonResult with agreement score, conflicts, and recommendation
    """
    p = _get_provider(provider, api_key, model)

    user_msg = (
        f'RESPONSE A:\n"""\n{response_a}\n"""\n\n'
        f'RESPONSE B:\n"""\n{response_b}\n"""\n\n'
        f"Respond ONLY in valid JSON."
    )

    raw = p.complete(SYSTEM_PROMPT, user_msg)
    data = _parse_json(raw)

    ra = data.get("response_a", {})
    rb = data.get("response_b", {})

    return ComparisonResult(
        response_a_compass=Compass(
            T=max(0, min(1, float(ra.get("T", 0.5)))),
            I=max(0, min(1, float(ra.get("I", 0.5)))),
            F=max(0, min(1, float(ra.get("F", 0.1)))),
            label="Response A",
        ),
        response_b_compass=Compass(
            T=max(0, min(1, float(rb.get("T", 0.5)))),
            I=max(0, min(1, float(rb.get("I", 0.5)))),
            F=max(0, min(1, float(rb.get("F", 0.1)))),
            label="Response B",
        ),
        agreement=float(data.get("agreement", 0.5)),
        agreements=data.get("agreements", []),
        conflicts=data.get("conflicts", []),
        combined_zone=data.get("combined_zone", "ambiguity"),
        more_honest=data.get("more_honest", "neither"),
        recommendation=data.get("recommendation", "investigate"),
    )
