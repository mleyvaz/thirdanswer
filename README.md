# thirdanswer

**The Third Answer** — evaluate AI uncertainty with T,I,F (Truth, Indeterminacy, Falsity).

Based on the book *"The Third Answer: Why AI Doesn't Know What It Doesn't Know — And How Ancient Logic Can Fix It"* by Leyva-Vazquez & Smarandache (2026).

## Install

```bash
pip install thirdanswer
```

For LLM analysis (free via Groq):
```bash
pip install thirdanswer[groq]
```

## Quick Start

### Without LLM (pure logic)

```python
from thirdanswer import Compass

# Create a reading
c = Compass(T=0.7, I=0.4, F=0.5)
print(c.zone)              # "contradiction"
print(c.zone_emoji)        # "🟠"
print(c.confidence)        # 0.0
print(c.is_paraconsistent) # True (T+F=1.2 > 1)
print(c.should_abstain)    # False
print(c.zone_action)       # "Investigate both sides..."
```

### Analyze any text (with Groq, free)

```python
from thirdanswer import analyze

r = analyze(
    "Coffee is definitively good for your health",
    provider="groq",
    api_key="gsk_..."  # Free at console.groq.com
)
print(r)           # Analysis(T=0.55, I=0.30, F=0.45) → 🟠 Contradiction
print(r.zone)      # "contradiction"
print(r.errors)    # ["confident_ignorance"] if detected
print(r.honest)    # Rewritten version with uncertainty

# Claim-by-claim
for claim in r.claims:
    print(f"{claim.zone_emoji} {claim.text[:50]}  T={claim.T}")
```

### Ask honest questions

```python
from thirdanswer import ask

r = ask("Is intermittent fasting healthy?", provider="groq", api_key="gsk_...")
print(r.answer)
print(r.what_i_dont_know)  # "Long-term effects beyond 2 years..."
print(r.zone)              # "contradiction"
```

### Compare two AI responses

```python
from thirdanswer import compare

diff = compare(chatgpt_response, claude_response, provider="groq", api_key="gsk_...")
print(f"Agreement: {diff.agreement:.0%}")
print(f"Conflicts: {diff.conflicts}")
print(f"More honest: {diff.more_honest}")
```

### Decompose text into claims

```python
from thirdanswer import decompose

claims = decompose("Long AI response...", provider="groq", api_key="gsk_...")
for c in claims:
    print(f"{c.zone_emoji} T={c.T} I={c.I} F={c.F} | {c.text}")
```

## The Four Zones

| Zone | Condition | Action |
|------|-----------|--------|
| 🟢 Consensus | T high, I low, F low | **Trust** |
| 🟡 Ambiguity | I high | **Investigate** |
| 🟠 Contradiction | T high AND F high | **Explore both sides** |
| ⚫ Ignorance | All low or I overwhelming | **Stop** |

## Providers

| Provider | Cost | API Key | Install |
|----------|------|---------|---------|
| Groq | Free (~30 req/min) | [console.groq.com](https://console.groq.com) | `pip install thirdanswer[groq]` |
| Ollama | Free (local) | None | [ollama.com](https://ollama.com) |

`Compass` works without any provider — pure logic, no API calls.

## Links

- [The Third Answer App](https://the-third-answer.streamlit.app)
- [Prompt Templates](https://github.com/mleyvaz/thirdanswer-prompts)
- [Book](https://thethirdanswer.com) *(coming soon)*
- [ORCID: M. Leyva-Vazquez](https://orcid.org/0000-0001-5401-0018)

## License

MIT
