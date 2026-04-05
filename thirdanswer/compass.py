"""
Compass — Core T,I,F logic. No LLM needed, pure neutrosophic math.

Usage:
    from thirdanswer import Compass
    c = Compass(T=0.7, I=0.4, F=0.5)
    c.zone            # "contradiction"
    c.confidence      # 0.0
    c.is_paraconsistent  # True
"""

from dataclasses import dataclass, field
from typing import Optional


ZONES = {
    "consensus": {
        "name": "Consensus",
        "emoji": "\U0001f7e2",
        "action": "Trust — but verify sources for critical decisions",
        "description": "Evidence supports the claim. Little uncertainty. Negligible counter-evidence.",
    },
    "consensus_against": {
        "name": "Consensus (Against)",
        "emoji": "\U0001f534",
        "action": "Reject — the evidence contradicts this claim",
        "description": "Strong evidence against the claim. Likely false or misleading.",
    },
    "ambiguity": {
        "name": "Ambiguity",
        "emoji": "\U0001f7e1",
        "action": "Investigate — the evidence is insufficient",
        "description": "High indeterminacy. The data doesn't exist yet or is too sparse.",
    },
    "contradiction": {
        "name": "Contradiction",
        "emoji": "\U0001f7e0",
        "action": "Investigate both sides — the evidence conflicts",
        "description": "Both T and F are significant. Evidence supports AND contradicts.",
    },
    "ignorance": {
        "name": "Ignorance",
        "emoji": "\u26ab",
        "action": "Stop — the AI is operating in the dark",
        "description": "No meaningful information. The output is not grounded in anything.",
    },
}


def _classify_zone(t: float, i: float, f: float) -> str:
    """Classify a T,I,F triple into one of the Four Zones."""
    if i > 0.5:
        if t < 0.3 and f < 0.3:
            return "ignorance"
        return "ambiguity"
    if t > 0.5 and f > 0.4:
        return "contradiction"
    if t > 0.5 and i < 0.35 and f < 0.3:
        return "consensus"
    if f > 0.5 and t < 0.3:
        return "consensus_against"
    if t < 0.3 and f < 0.3 and i < 0.3:
        return "ignorance"
    if i > 0.35:
        return "ambiguity"
    if t > 0.3 and f > 0.3:
        return "contradiction"
    return "ambiguity"


@dataclass
class Compass:
    """
    The epistemic compass. Three independent needles.

    Args:
        T: Truth — degree of evidence supporting the claim [0, 1]
        I: Indeterminacy — degree of genuine uncertainty [0, 1]
        F: Falsity — degree of evidence contradicting the claim [0, 1]

    T + I + F do NOT need to equal 1. They are independent.
    """

    T: float
    I: float
    F: float
    label: Optional[str] = None

    def __post_init__(self):
        for name, val in [("T", self.T), ("I", self.I), ("F", self.F)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {val}")

    @property
    def zone(self) -> str:
        """The epistemic zone: consensus, ambiguity, contradiction, ignorance."""
        return _classify_zone(self.T, self.I, self.F)

    @property
    def zone_name(self) -> str:
        return ZONES[self.zone]["name"]

    @property
    def zone_emoji(self) -> str:
        return ZONES[self.zone]["emoji"]

    @property
    def zone_action(self) -> str:
        return ZONES[self.zone]["action"]

    @property
    def zone_description(self) -> str:
        return ZONES[self.zone]["description"]

    @property
    def confidence(self) -> float:
        """C(sigma) = max(0, T - I - F)"""
        return max(0.0, self.T - self.I - self.F)

    @property
    def is_paraconsistent(self) -> bool:
        """True when T + F > 1 — contradictory evidence coexists."""
        return (self.T + self.F) > 1.0

    @property
    def tf_sum(self) -> float:
        """T + F — when > 1, paraconsistent state."""
        return self.T + self.F

    @property
    def should_abstain(self) -> bool:
        """True when the responsible action is to not decide."""
        return self.zone in ("ignorance", "ambiguity") or self.I > 0.6

    def to_dict(self) -> dict:
        return {
            "T": self.T,
            "I": self.I,
            "F": self.F,
            "zone": self.zone,
            "zone_name": self.zone_name,
            "zone_emoji": self.zone_emoji,
            "action": self.zone_action,
            "confidence": round(self.confidence, 4),
            "is_paraconsistent": self.is_paraconsistent,
            "should_abstain": self.should_abstain,
            "label": self.label,
        }

    def __repr__(self) -> str:
        p = " PARACONSISTENT" if self.is_paraconsistent else ""
        lbl = f' "{self.label}"' if self.label else ""
        return (
            f"Compass(T={self.T}, I={self.I}, F={self.F}) "
            f"→ {self.zone_emoji} {self.zone_name}{p}{lbl}"
        )
