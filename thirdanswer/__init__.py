"""
thirdanswer — The Third Answer framework for evaluating AI uncertainty.

Why AI doesn't know what it doesn't know — and how to find out.

Basic usage (no LLM needed):
    from thirdanswer import Compass
    c = Compass(T=0.7, I=0.4, F=0.5)
    print(c.zone)  # "contradiction"

With LLM (Groq, free):
    from thirdanswer import analyze
    r = analyze("AI text here", provider="groq", api_key="gsk_...")
    print(r.zone, r.T, r.I, r.F)

Book: "The Third Answer" by Leyva-Vazquez & Smarandache (2026)
"""

__version__ = "0.2.0"
__author__ = "Maikel Yelandi Leyva-Vazquez"

from .compass import Compass
from .analyzer import analyze, decompose, AnalysisResult, Claim
from .honest import ask, HonestResponse
from .compare import compare, ComparisonResult
from .errors import AIError, ERROR_TYPES

__all__ = [
    "Compass",
    "analyze",
    "decompose",
    "ask",
    "compare",
    "AnalysisResult",
    "HonestResponse",
    "ComparisonResult",
    "Claim",
    "AIError",
    "ERROR_TYPES",
]
