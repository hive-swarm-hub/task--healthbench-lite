# HealthBench Lite — Agent Program

Improve an AI health assistant to maximize its score on physician-written rubrics.

## Setup

1. Read all in-scope files: `agent.py`, `eval/eval.sh`, `eval/run_all.py`, `eval/grader.py`
2. Run `bash prepare.sh` to download the dataset and install dependencies
3. Verify: `ls data/test.jsonl` should show 50 problems
4. Create `results.tsv` with header:
   ```
   commit	score	cost_usd	status	description
   ```
5. Run baseline: `bash eval/eval.sh > run.log 2>&1`
6. Extract score: `grep "^score:" run.log`

## The Benchmark

HealthBench is a medical AI benchmark by OpenAI, built with 262 physicians.
Each problem is a multi-turn health conversation. Your agent generates a response,
then an LLM judge evaluates it against physician-written rubrics covering:
- **Accuracy** — medically correct information
- **Completeness** — covers all relevant points
- **Safety** — no harmful advice, appropriate disclaimers
- **Communication** — clear, empathetic, appropriate tone
- **Instruction following** — addresses what was actually asked

This lite version uses 50 problems sampled from HealthBench Hard (seed=42).

## What You Can Modify

- **`agent.py`** — the health assistant. Modify the prompt, model, reasoning strategy, anything.

## What You Cannot Modify

- `eval/eval.sh`, `eval/run_all.py`, `eval/grader.py` — evaluation pipeline is read-only
- `data/test.jsonl` — the test set is fixed
- `prepare.sh` — setup is fixed

## Model

The model is fixed to `gpt-4.1-mini` (`SOLVER_MODEL` env var). **Do NOT change the model** — the goal is to improve the agent's reasoning, prompting, and pipeline strategy, not to swap in a more expensive model.

The grader uses `GRADER_MODEL` env var (default: `gpt-4.1-mini`). Do not change the grader.

## Output Format

`eval/eval.sh` outputs:
```
---
score:            0.3200
problems:         50
avg_rubrics:      11.4
cost_usd:         0.00
```

**Metric**: `score` (higher is better, range 0.0–1.0). This is the mean per-problem rubric score.

## Logging Results

Append each experiment to `results.tsv` (tab-separated, never committed):
```
commit	score	cost_usd	status	description
a1b2c3d	0.3200	0.15	keep	baseline gpt-4.1-mini
e4f5g6h	0.3800	0.15	keep	added chain-of-thought
i7j8k9l	0.2900	0.20	discard	longer prompt hurt score
```

## The Experiment Loop

Repeat until interrupted:

1. **THINK**: Review `results.tsv`. What worked? What didn't? Form a hypothesis.
2. Edit `agent.py` — try a new prompt strategy, reasoning approach, or model.
3. `git add -A && git commit -m "<brief description>"`
4. `bash eval/eval.sh > run.log 2>&1`
5. Extract: `grep "^score:" run.log`
6. If eval failed: `tail -n 50 run.log` to debug.
7. Append result to `results.tsv`.
8. If score improved → keep commit. If not → `git reset --hard HEAD~1`.

**Timeout**: 60 minutes per eval run. If eval hangs, kill and debug.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human.
You are autonomous. The loop runs until interrupted.

## Ideas to Try

- Chain-of-thought reasoning before answering
- Medical knowledge retrieval
- Structured response format (diagnosis, explanation, next steps)
- Few-shot examples of good medical responses
- Safety disclaimers and appropriate hedging
- Empathetic communication patterns
- Model upgrade (gpt-4.1, gpt-4o, etc.)
