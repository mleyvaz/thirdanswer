"""
ask() — Generate honest responses WITH T,I,F built in.

Usage:
    from thirdanswer import ask
    r = ask("Is intermittent fasting healthy?", provider="groq", api_key="gsk_...")
    print(r.answer)
    print(r.what_i_dont_know)
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .compass import Compass, _classify_zone
from .analyzer import Claim, _parse_json, _get_provider


SYSTEM_PROMPT = """You answer questions honestly using the Third Answer framework.

For every response:
1. Answer the question accurately
2. For each key claim, indicate certainty:
   - "well-established" (T > 0.8, I < 0.15)
   - "debated" (T > 0.5, F > 0.2)
   - "uncertain" (I > 0.5)
   - "contradicted" (T > 0.3 AND F > 0.3)
   - "unreliable" (I > 0.6, T < 0.3)
3. State explicitly what you DON'T know

Respond in JSON:
{
  "answer": "detailed answer with inline uncertainty markers",
  "T": 0.0, "I": 0.0, "F": 0.0,
  "zone": "consensus|ambiguity|contradiction|ignorance",
  "zone_reason": "why",
  "claims": [
    {"claim": "...", "certainty": "well-established|debated|uncertain|contradicted|unreliable", "T": 0.0, "I": 0.0, "F": 0.0}
  ],
  "recommendation": "trust|investigate|consult_expert|do_not_use",
  "what_i_dont_know": "explicit statement of unknowns"
}"""


@dataclass
class HonestResponse:
    """An AI response that includes epistemic self-assessment."""
    answer: str
    T: float
    I: float
    F: float
    zone: str
    zone_reason: str
    claims: List[Claim]
    recommendation: str
    what_i_dont_know: str
    provider_name: str = ""

    @property
    def compass(self) -> Compass:
        return Compass(T=self.T, I=self.I, F=self.F)

    @property
    def zone_emoji(self) -> str:
        return self.compass.zone_emoji

    @property
    def zone_name(self) -> str:
        return self.compass.zone_name

    @property
    def confidence(self) -> float:
        return self.compass.confidence

    @property
    def is_paraconsistent(self) -> bool:
        return self.compass.is_paraconsistent

    def __repr__(self) -> str:
        preview = self.answer[:80] + "..." if len(self.answer) > 80 else self.answer
        return (
            f"HonestResponse({self.zone_emoji} {self.zone_name} | "
            f"T={self.T} I={self.I} F={self.F})\n"
            f"  {preview}"
        )


def ask(
    question: str,
    provider: str = "groq",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    domain: str = "general",
) -> HonestResponse:
    """
    Ask a question and get an honest response WITH T,I,F assessment.

    The AI answers your question AND tells you what it doesn't know.

    Args:
        question: Your question
        provider: "groq" (free) or "ollama" (local)
        api_key: API key (not needed for ollama)
        model: Model override
        domain: Context domain (general, medicine, legal, etc.)

    Returns:
        HonestResponse with answer, T, I, F, zone, and what_i_dont_know
    """
    p = _get_provider(provider, api_key, model)

    user_msg = f"Domain: {domain}\nQuestion: {question}\n\nRespond in JSON."
    raw = p.complete(SYSTEM_PROMPT, user_msg)
    data = _parse_json(raw)

    for key in ["T", "I", "F"]:
        data.setdefault(key, 0.5)
        data[key] = max(0.0, min(1.0, float(data[key])))

    claims = [
        Claim(
            text=c.get("claim", ""),
            T=max(0.0, min(1.0, float(c.get("T", 0.5)))),
            I=max(0.0, min(1.0, float(c.get("I", 0.5)))),
            F=max(0.0, min(1.0, float(c.get("F", 0.1)))),
            note=c.get("certainty", ""),
        )
        for c in data.get("claims", [])
    ]

    return HonestResponse(
        answer=data.get("answer", ""),
        T=data["T"],
        I=data["I"],
        F=data["F"],
        zone=data.get("zone", _classify_zone(data["T"], data["I"], data["F"])),
        zone_reason=data.get("zone_reason", ""),
        claims=claims,
        recommendation=data.get("recommendation", "investigate"),
        what_i_dont_know=data.get("what_i_dont_know", ""),
        provider_name=p.name,
    )
