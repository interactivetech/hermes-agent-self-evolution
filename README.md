# 🧬 Hermes Agent Self-Evolution

**Evolutionary self-improvement for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Hermes Agent Self-Evolution uses DSPy + GEPA (Genetic-Pareto Prompt Evolution) to automatically evolve and optimize Hermes Agent's skills, tool descriptions, system prompts, and code — producing measurably better versions through reflective evolutionary search.

**No GPU training required.** Everything operates via API calls — mutating text, evaluating results, and selecting the best variants. ~$2-10 per optimization run.

## How It Works

```
Read current skill/prompt/tool ──► Generate eval dataset
                                        │
                                        ▼
                                   GEPA Optimizer ◄── Execution traces
                                        │                    ▲
                                        ▼                    │
                                   Candidate variants ──► Evaluate
                                        │
                                   Constraint gates (tests, size limits, benchmarks)
                                        │
                                        ▼
                                   Best variant ──► PR against hermes-agent
```

GEPA reads execution traces to understand *why* things fail (not just that they failed), then proposes targeted improvements. ICLR 2026 Oral, MIT licensed.

## Quick Start

```bash
# Install this repo
git clone https://github.com/NousResearch/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution

# Create a local Python 3.11 virtualenv for this project
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# If you expect to use DSPy's MIPROv2 fallback optimizer, also install optuna.
# zsh requires quoting brackets in extras syntax:
uv pip install 'dspy[optuna]'

# Point at your Hermes source checkout
# Current local setup:
#   Hermes repo:  ~/Documents/2026_research_projects/hermes-dev
#   Hermes home:  ~/.hermes-dev
export HERMES_AGENT_REPO=~/Documents/2026_research_projects/hermes-dev
export HERMES_HOME=~/.hermes-dev

# Set credentials for the model provider you plan to use if needed.
# This is only required when you are not already using a custom provider
# from $HERMES_HOME/config.yaml.
export OPENAI_API_KEY=your_key_here

# Optional sanity check
test -d "$HERMES_AGENT_REPO/.git" && echo "Hermes repo found"
test -f "$HERMES_HOME/config.yaml" && echo "Hermes dev config found"
test -n "$OPENAI_API_KEY" && echo "OpenAI key found (optional if using Hermes custom provider)"

# Evolve a skill (synthetic eval data)
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source synthetic

# Or use real session history from Claude Code, Copilot, and Hermes
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source sessiondb
```

If you use a different LiteLLM provider, pass provider-qualified model names such as
`--eval-model anthropic/claude-sonnet-4` and export that provider's API key first.

If your Hermes config already points at a custom OpenAI-compatible endpoint such as vLLM,
this repo will automatically reuse the active provider from `$HERMES_HOME/config.yaml`.
You can also override it explicitly with `--base-url`, `--api-key`, `--eval-model`,
and `--optimizer-model`.

## Current State

Phase 1 skill evolution is runnable with DSPy + GEPA against local Hermes skills,
including Hermes-configured OpenAI-compatible endpoints such as vLLM.

Current behavior and caveats:

- The active optimization path is still GEPA over a DSPy skill wrapper.
- Synthetic datasets are generated from the target skill text and are useful for
  plumbing validation, but they are not grounded in a real repository/task fixture.
- Result scores currently come from a lightweight rubric/keyword-overlap metric, so
  runs can overfit to rubric wording rather than improve real task performance.
- A run may improve the DSPy program behavior without producing a materially changed
  `SKILL.md` artifact; interpret output metrics accordingly.
- Output artifacts are written under `output/<skill>/<timestamp>/` for manual review.

## What It Optimizes

| Phase | Target | Engine | Status |
|-------|--------|--------|--------|
| **Phase 1** | Skill files (SKILL.md) | DSPy + GEPA | ✅ Implemented |
| **Phase 2** | Tool descriptions | DSPy + GEPA | 🔲 Planned |
| **Phase 3** | System prompt sections | DSPy + GEPA | 🔲 Planned |
| **Phase 4** | Tool implementation code | Darwinian Evolver | 🔲 Planned |
| **Phase 5** | Continuous improvement loop | Automated pipeline | 🔲 Planned |

## Engines

| Engine | What It Does | License |
|--------|-------------|---------|
| **[DSPy](https://github.com/stanfordnlp/dspy) + [GEPA](https://github.com/gepa-ai/gepa)** | Reflective prompt evolution — reads execution traces, proposes targeted mutations | MIT |
| **[Darwinian Evolver](https://github.com/imbue-ai/darwinian_evolver)** | Code evolution with Git-based organisms | AGPL v3 (external CLI only) |

## Guardrails

Every evolved variant must pass:
1. **Full test suite** — `pytest tests/ -q` must pass 100%
2. **Size limits** — Skills ≤15KB, tool descriptions ≤500 chars
3. **Caching compatibility** — No mid-conversation changes
4. **Semantic preservation** — Must not drift from original purpose
5. **PR review** — All changes go through human review, never direct commit

## Full Plan

See [PLAN.md](PLAN.md) for the complete architecture, evaluation data strategy, constraints, benchmarks integration, and phased timeline.

## License

MIT — © 2026 Nous Research
