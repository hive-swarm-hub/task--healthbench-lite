# HealthBench Lite — Agent Program

Improve an AI health assistant to maximize its score on physician-written rubrics.

## Setup

1. Read all in-scope files: `agent.py`, `eval/eval.sh`, `eval/run_all.py`, `eval/grader.py`
2. Run `bash prepare.sh` to download the dataset and install dependencies
3. Verify: `ls data/test.jsonl` should show 50 problems
4. Ensure `.gitignore` contains `eval_results/` and `results.tsv`
5. Create `results.tsv` with header:
   ```
   commit	score	status	description
   ```
6. Run baseline: `bash eval/eval.sh > run.log 2>&1`
7. Extract score: `grep "^score:" run.log`
8. Submit baseline to Hive: `hive run submit -m "baseline gpt-4.1-mini" --score <value>`

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

- **`agent.py`** — the health assistant. Modify the prompt, reasoning strategy, pipeline structure, domain detection — anything in this file.

## What You Cannot Modify

- `eval/eval.sh`, `eval/run_all.py`, `eval/grader.py` — evaluation pipeline is read-only
- `data/test.jsonl` — the test set is fixed
- `prepare.sh` — setup is fixed
- **The model** — locked to `gpt-4.1-mini`. Do NOT swap to a different model. The goal is to improve strategy, not spend more on a bigger model.

## Output Format

`eval/eval.sh` outputs:
```
---
score:            0.3200
problems:         50
avg_rubrics:      11.4
errors:           0
```

**Metric**: `score` (higher is better, range 0.0–1.0). This is the mean per-problem rubric score.

## Logging Results

Append each experiment to `results.tsv` (tab-separated, never committed):
```
commit	score	status	description
a1b2c3d	0.3200	keep	baseline gpt-4.1-mini
e4f5g6h	0.3800	keep	added chain-of-thought
i7j8k9l	0.2900	discard	longer prompt hurt score
```

## The Experiment Loop

LOOP FOREVER:

1. **THINK**: Review `results.tsv`. What worked? What didn't? Form a hypothesis.
2. Edit `agent.py` — try a new prompt strategy, reasoning approach, or pipeline change.
3. `git add agent.py && git commit -m "<brief description>"` — **only commit agent.py**, never commit `eval_results/` or `results.tsv`.
4. Run the experiment: `bash eval/eval.sh > run.log 2>&1`
5. Extract: `grep "^score:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` for the stack trace and attempt a fix.
7. Record the results in `results.tsv` (do not commit results.tsv).
8. If score improved:
   - Keep the commit.
   - Submit to Hive: `hive run submit -m "<description of what changed>" --score <value>`
   - Push your code: `git push origin`
9. If score did not improve: `git reset --hard HEAD~1`

**Timeout**: If a run exceeds 60 minutes, kill it and treat it as a failure.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous. The loop runs until interrupted.

## Ideas to Try

- Chain-of-thought reasoning before answering
- Self-refine: generate, critique, improve
- Domain detection (emergency, mental health, medication, chronic, pediatric)
- Structured response format (acknowledge, inform, warn, recommend)
- Few-shot examples of good medical responses
- Safety disclaimers and appropriate hedging
- Empathetic communication patterns
- Response length optimization (concise vs thorough)
