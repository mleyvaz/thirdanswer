"""
analyze() — Evaluate any AI-generated text with T,I,F.

Usage:
    from thirdanswer import analyze
    r = analyze("Coffee is good for health", provider="groq", api_key="gsk_...")
    print(r.T, r.I, r.F, r.zone)
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .compass import Compass, _classify_zone
from .providers.base import Provider


SYSTEM_PROMPT = """You are an epistemic analysis engine implementing the Third Answer framework.

For ANY text given to you, analyze its epistemic quality using three INDEPENDENT dimensions:
- T (Truth): degree of evidence supporting the claim [0.0 to 1.0]
- I (Indeterminacy): degree of genuine uncertainty [0.0 to 1.0]
- F (Falsity): degree of evidence contradicting the claim [0.0 to 1.0]

T + I + F do NOT need to equal 1. They are independent.

Classify into zones:
- "consensus": T > 0.5, I < 0.35, F < 0.3
- "ambiguity": I > 0.5 or (I > 0.35 and T < 0.6)
- "contradiction": T > 0.3 AND F > 0.3
- "ignorance": T < 0.3, F < 0.3, or I overwhelming

Identify error types: fabrication, distortion, conflation, confident_ignorance

Respond ONLY in valid JSON:
{
  "T": 0.0, "I": 0.0, "F": 0.0,
  "zone": "consensus|ambiguity|contradiction|ignorance",
  "zone_reason": "why this zone",
  "error_types": [],
  "error_explanation": "",
  "honest_version": "text rewritten with uncertainty markers",
  "claims": [
    {"claim": "text", "T": 0.0, "I": 0.0, "F": 0.0, "note": "reason"}
  ],
  "recommendation": "trust|investigate|consult_expert|do_not_use"
}"""


@dataclass
class Claim:
    """A single claim extracted from text with its T,I,F assessment."""
    text: str
    T: float
    I: float
    F: float
    note: str = ""

    @property
    def compass(self) -> Compass:
        return Compass(T=self.T, I=self.I, F=self.F, label=self.text[:50])

    @property
    def zone(self) -> str:
        return self.compass.zone

    @property
    def zone_emoji(self) -> str:
        return self.compass.zone_emoji


@dataclass
class AnalysisResult:
    """Result of analyzing text with the Third Answer framework."""
    T: float
    I: float
    F: float
    zone: str
    zone_reason: str
    error_types: List[str]
    error_explanation: str
    honest_version: str
    claims: List[Claim]
    recommendation: str
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
    def zone_action(self) -> str:
        return self.compass.zone_action

    @property
    def confidence(self) -> float:
        return self.compass.confidence

    @property
    def is_paraconsistent(self) -> bool:
        return self.compass.is_paraconsistent

    @property
    def has_errors(self) -> bool:
        return len(self.error_types) > 0

    def label(self) -> str:
        """Generate the Epistemic Nutrition Label for this analysis."""
        bar_t = "\u2588" * int(self.T * 8) + "\u2591" * (8 - int(self.T * 8))
        bar_i = "\u2588" * int(self.I * 8) + "\u2591" * (8 - int(self.I * 8))
        bar_f = "\u2588" * int(self.F * 8) + "\u2591" * (8 - int(self.F * 8))
        para = " PARACONSISTENT" if self.is_paraconsistent else ""
        errs = f"\n  Errors: {', '.join(self.error_types)}" if self.has_errors else ""
        return (
            f"\u250c{'─'*42}\u2510\n"
            f"\u2502  EPISTEMIC NUTRITION LABEL{' '*15}\u2502\n"
            f"\u2502{'═'*42}\u2502\n"
            f"\u2502  Truth (T)          {bar_t} {self.T:.2f}\u2502\n"
            f"\u2502  Indeterminacy (I)  {bar_i} {self.I:.2f}\u2502\n"
            f"\u2502  Falsity (F)        {bar_f} {self.F:.2f}\u2502\n"
            f"\u2502{'─'*42}\u2502\n"
            f"\u2502  Zone: {self.zone_emoji} {self.zone_name:.<28s}\u2502\n"
            f"\u2502  Action: {self.compass.zone_action[:32]:.<32s}\u2502\n"
            f"\u2502  Confidence: {self.confidence:.2f}{para:.<27s}\u2502\n"
            f"\u2502  Recommendation: {self.recommendation:.<24s}\u2502\n"
            f"\u2502{'─'*42}\u2502\n"
            f"\u2502  Powered by thirdanswer{' '*18}\u2502\n"
            f"\u2514{'─'*42}\u2518"
            f"{errs}"
        )

    def __repr__(self) -> str:
        errs = f" ERRORS: {self.error_types}" if self.has_errors else ""
        return (
            f"Analysis(T={self.T}, I={self.I}, F={self.F}) "
            f"→ {self.zone_emoji} {self.zone_name} | {self.recommendation}{errs}"
        )


def _parse_json(raw: str) -> dict:
    """Extract JSON from LLM response."""
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        raw = match.group()
    return json.loads(raw)


def _get_provider(provider: str, api_key: Optional[str] = None,
                  model: Optional[str] = None) -> Provider:
    """Instantiate a provider by name."""
    if provider == "groq":
        from .providers.groq import GroqProvider
        return GroqProvider(api_key=api_key, model=model or "llama-3.3-70b-versatile")
    elif provider == "ollama":
        from .providers.ollama import OllamaProvider
        return OllamaProvider(model=model or "llama3.1")
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'groq' or 'ollama'.")


def analyze(
    text: str,
    provider: str = "groq",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    context: str = "",
) -> AnalysisResult:
    """
    Analyze any text using the Third Answer T,I,F framework.

    Args:
        text: The AI-generated (or any) text to evaluate
        provider: "groq" (free, needs key) or "ollama" (local, free)
        api_key: API key for the provider (not needed for ollama)
        model: Model override (default depends on provider)
        context: Optional domain context (e.g., "medicine", "legal")

    Returns:
        AnalysisResult with T, I, F, zone, claims, errors, honest version
    """
    p = _get_provider(provider, api_key, model)

    user_msg = f"Analyze this text using the Third Answer framework:\n\n"
    if context:
        user_msg += f"Context/domain: {context}\n\n"
    user_msg += f'TEXT:\n"""\n{text}\n"""\n\nRespond ONLY in valid JSON.'

    raw = p.complete(SYSTEM_PROMPT, user_msg)
    data = _parse_json(raw)

    # Defaults and clamping
    for key in ["T", "I", "F"]:
        data.setdefault(key, 0.5)
        data[key] = max(0.0, min(1.0, float(data[key])))

    claims = [
        Claim(
            text=c.get("claim", ""),
            T=max(0.0, min(1.0, float(c.get("T", 0.5)))),
            I=max(0.0, min(1.0, float(c.get("I", 0.5)))),
            F=max(0.0, min(1.0, float(c.get("F", 0.1)))),
            note=c.get("note", ""),
        )
        for c in data.get("claims", data.get("key_claims", []))
    ]

    return AnalysisResult(
        T=data["T"],
        I=data["I"],
        F=data["F"],
        zone=data.get("zone", _classify_zone(data["T"], data["I"], data["F"])),
        zone_reason=data.get("zone_reason", ""),
        error_types=data.get("error_types", []),
        error_explanation=data.get("error_explanation", ""),
        honest_version=data.get("honest_version", text),
        claims=claims,
        recommendation=data.get("recommendation", "investigate"),
        provider_name=p.name,
    )


def decompose(
    text: str,
    provider: str = "groq",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Claim]:
    """
    Decompose text into individual claims, each with T,I,F.

    Args:
        text: Text containing multiple claims
        provider: "groq" or "ollama"
        api_key: API key (not needed for ollama)

    Returns:
        List of Claim objects
    """
    result = analyze(text, provider=provider, api_key=api_key, model=model)
    return result.claims
