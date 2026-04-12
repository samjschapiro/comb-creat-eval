# AI Operations Protocol

**The philosophy: nothing I (the AI) do can be trusted without your verification.**
Assume every operation I report as "done" or "safe" may have hidden side effects
(orphan processes, wrong configs, unexpected API calls). This document lays out
how to verify every state change I make and how I operate going forward.

## The Incident (2026-04-12)

A python process spawned ~7 hours earlier (a "smoke test" I launched with
`uv run python ... | head -20`) kept running after the `head` pipe closed,
because `head` only closed the pipe — not the upstream process. Subsequent
`TaskStop` calls only stopped tasks explicitly registered as background tasks;
the orphan was never registered. Result: an unnoticed background process made
API calls against the old config (wrong temperatures, wrong model order) for
~7 hours, including expensive frontier models the user didn't authorize.

**Lesson**: pipes (`| head`, `| tee`, etc.) can leave zombie parents. Never
use them for process-launch commands. Always register background tasks with
`run_in_background: true` so `TaskStop` can reach them.

## User-Facing Safety Commands

### Check what's running right now
```bash
bash scripts/safety/status.sh
```
Shows:
1. Any python/eval processes still alive (with PIDs)
2. Files modified in the last 60 seconds (live writes)
3. Files modified in the last 10 minutes (recent activity)
4. Eval progress per model
5. Active network connections to LLM APIs

### Kill everything (nuclear option)
```bash
bash scripts/safety/kill_all.sh
```
Finds and SIGTERMs all `python src/dat_eval/...`, `python src/comb_eval/...`,
and `uv run python src/...` processes. Escalates to SIGKILL if anything
survives. Safe to run anytime — it will only kill processes related to this
repo.

### Estimate spend so far
```bash
uv run python scripts/safety/cost_tracker.py
```
Walks `data/dat_eval/run_v1/` and estimates USD spend from token counts in
saved response files. **This is an estimate** — for truth check
https://openrouter.ai/activity.

## Rules I Follow Going Forward

### 1. NEVER launch a long-running process via a pipe.

❌ **Wrong** (what caused the incident):
```
uv run python src/dat_eval/scripts/run_evals.py ... | head -20
```
The `head -20` exits after 20 lines; upstream python keeps running.

✅ **Right**: use `run_in_background: true` in the Bash tool so the task is
registered and killable via `TaskStop`. For debug checks, use `--debug` flag
that limits scope (1 model, 1 trial) and completes quickly.

### 2. NEVER launch a new eval run without first verifying nothing else is running.

Before any `run_evals.py` or similar long-running launch, I must first:
```bash
ps aux | grep -iE "(python.*src/(dat_eval|comb_eval))" | grep -v grep
```
and report the result. If anything is running, kill it first.

### 3. ALWAYS report actual current state, not expected state.

After `TaskStop`, I verify the process is actually dead via `ps`. I don't
trust the fact that the task notification says "completed" or "stopped."

### 4. ALWAYS surface cost-relevant decisions explicitly.

Before starting any eval run that queries paid APIs:
- State the estimated total cost
- State the model list and order (expensive → cheap or vice versa)
- State the number of API calls per model
- Wait for explicit user approval

Never make "while you're not looking" changes to API call plans.

### 5. ALWAYS log significant operations to the operations log.

Every significant action (launching a run, killing a process, modifying a
config that affects spend, deleting response files) should be appended to
[ai_operations_log.md](memos/ai_operations_log.md).

## What You Should Do

**When you come back to the project:**
1. Run `bash scripts/safety/status.sh` — confirm no rogue processes
2. Run `uv run python scripts/safety/cost_tracker.py` — estimate spend
3. Check https://openrouter.ai/activity — truth of spend
4. Read the last entries in [ai_operations_log.md](memos/ai_operations_log.md)

**If something feels wrong:**
1. `bash scripts/safety/kill_all.sh` — nuke everything related to this repo
2. Re-run `status.sh` to confirm

**When I ask to run something expensive:**
1. Make me state expected cost, models, and call count before approving
2. Verify the status shows no rogue processes first

## What This Protocol Can't Do

- It can't prevent me from introducing subtle bugs that waste money on
  retries, broken prompts, or redundant calls
- It can't detect spend until *after* the API call happens
- It depends on me reliably reporting what I'm about to do, which I just
  proved I'm not always good at

Therefore: the most reliable signal is **your OpenRouter dashboard at
https://openrouter.ai/activity**. Check it frequently. It is the ground truth.
My status scripts are a proxy, not the real answer.
