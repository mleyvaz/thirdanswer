# thirdanswer

**Neutrosophic logic toolkit** — evaluate AI uncertainty and perform statistics beyond classical limits, using three independent dimensions: **T** (Truth), **I** (Indeterminacy), **F** (Falsity).

Based on *"The Third Answer"* by Leyva-Vazquez & Smarandache (2026) and *Neutrosophic Statistics* (Smarandache, 2022).

## What this library does that others cannot

| Capability | Classical stats | Interval stats | thirdanswer |
|---|---|---|---|
| Represent support AND opposition simultaneously | No | No | Yes (T and F independent) |
| Detect paraconsistency (T+F > 1) | No (P+Q=1) | No | Yes |
| Distinguish "no data" from "conflicting data" | No (both P=0.5) | No | Yes (Ignorance vs Contradiction) |
| Classify into 4 epistemic zones with actions | No | No | Yes |
| Partial sample membership (degree 0.6) | No (binary) | No | Yes |
| Reduce uncertainty by algebraic cancellation | N/A | No (always grows) | Yes |
| Evaluate AI text for hallucinations with T,I,F | N/A | N/A | Yes |

## Install

```bash
pip install thirdanswer
```

For LLM analysis (free via Groq):
```bash
pip install thirdanswer[groq]
```

## Part 1: Epistemic Compass (no LLM needed)

```python
from thirdanswer import Compass

c = Compass(T=0.7, I=0.4, F=0.5)
c.zone              # "contradiction"
c.confidence        # 0.0
c.is_paraconsistent # True (T+F=1.2 > 1)
c.zone_action       # "Investigate both sides..."
```

## Part 2: AI Text Analysis (with Groq, free)

```python
from thirdanswer import analyze, ask, compare

# Analyze any text
r = analyze("Coffee is good for health", provider="groq", api_key="gsk_...")
r.zone       # "contradiction"
r.label()    # Epistemic Nutrition Label

# Ask honest questions
r = ask("Is fasting healthy?", provider="groq", api_key="gsk_...")
r.what_i_dont_know  # "Long-term effects..."

# Compare two AI responses
diff = compare(response_a, response_b, provider="groq", api_key="gsk_...")
diff.more_honest    # "a" or "b"
```

## Part 3: Neutrosophic Statistics (v0.3.0)

Operations that classical and interval statistics **cannot** do.

### Neutrosophic Numbers (algebraic indeterminacy)

```python
from thirdanswer import NeutrosophicNumber, compare_uncertainty

N1 = NeutrosophicNumber(4, 2)   # 4 + 2I
N2 = NeutrosophicNumber(4, -2)  # 4 - 2I

# Average: neutrosophic cancels indeterminacy, intervals cannot
r = compare_uncertainty(N1, N2, "avg")
r["ns_uncertainty"]   # 0.0  (perfect cancellation!)
r["is_uncertainty"]   # 2.0  (intervals always grow)

# Product: NS is 4x more precise
r = compare_uncertainty(N1, N2, "mul")
r["ns_uncertainty"]   # 4
r["is_uncertainty"]   # 16
```

### Partial Membership Samples

```python
from thirdanswer import NeutrosophicSample, SampleElement

# Some students only partially belong to the cohort
sample = NeutrosophicSample([
    SampleElement(85, 1.0),   # full-time student
    SampleElement(72, 0.6),   # part-time
    SampleElement(65, 0.3),   # occasional attendee
])
sample.classical_mean()      # 74.0 (treats all equally)
sample.neutrosophic_mean()   # 78.7 (weights by membership — more accurate)
```

### Hesitant Sets (discrete, not interval)

```python
from thirdanswer import HesitantSet

# "The value might be 0.4, 7.9, or 41.5"
hs = HesitantSet([0.4, 7.9, 41.5])
hs.as_interval            # (0.4, 41.5) — uncertainty = 41.1
hs.mean                   # 16.6 — only 3 actual possibilities, not infinite
```

### Head-to-Head: Same P, Different Realities

```python
from thirdanswer import case_study_drug_efficacy

results = case_study_drug_efficacy()
# 4 drugs, ALL with P=0.55
# Classical: says "Support" for all 4 (accuracy: 0%)
# Neutrosophic: 4 different zones, 4 different actions (accuracy: 100%)
#   Drug A → Consensus (proceed)
#   Drug B → Ambiguity (need more data)
#   Drug C → Contradiction (studies disagree)
#   Drug D → Ignorance (no signal)
```

### Monte Carlo Proof

```python
from thirdanswer import monte_carlo_uncertainty

results = monte_carlo_uncertainty(n_trials=1000, seed=42)
# Addition:  NS wins 54.2%, IS wins 2.9%
# Multiply:  NS wins 68.4%, IS wins 9.1%
# NS is 1.38-1.54x more precise on average
```

### Run All Experiments

```python
from thirdanswer import run_all_experiments

all_results = run_all_experiments(seed=42)
# 6 experiments, fully reproducible, seeded
```

## The Four Zones

| Zone | Condition | Action |
|------|-----------|--------|
| Consensus | T high, I low, F low | **Trust** |
| Ambiguity | I high | **Investigate** |
| Contradiction | T high AND F high | **Explore both sides** |
| Ignorance | All low | **Stop** |

## Providers

| Provider | Cost | Install |
|----------|------|---------|
| Groq | Free (~30 req/min) | `pip install thirdanswer[groq]` |
| Ollama | Free (local) | [ollama.com](https://ollama.com) |

`Compass` and all `neutrostats` functions work without any provider — pure logic.

## Links

- [The Third Answer App](https://the-third-answer.streamlit.app)
- [Prompt Templates](https://github.com/mleyvaz/thirdanswer-prompts)
- [ORCID: M. Leyva-Vazquez](https://orcid.org/0000-0001-5401-0018)

## Citation

```bibtex
@software{thirdanswer,
  author = {Leyva-Vazquez, Maikel Y. and Smarandache, Florentin},
  title = {thirdanswer: Neutrosophic logic toolkit for AI uncertainty and statistics},
  year = {2026},
  url = {https://github.com/mleyvaz/thirdanswer}
}
```

## License

MIT
