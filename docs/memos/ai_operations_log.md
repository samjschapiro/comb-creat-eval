# AI Operations Log

Running log of significant actions I (the AI) take. Every entry should be
specific enough to audit — what I did, when, why, and what state change it
caused.

Format: `YYYY-MM-DD HH:MM | action | reason | side effects`

---

## 2026-04-12

### ~14:00 | Built safety infrastructure
- **Action**: Created `scripts/safety/{status.sh,kill_all.sh,cost_tracker.py}`,
  `docs/AI_OPERATIONS_PROTOCOL.md`, and this log.
- **Reason**: Orphan process incident (see below) revealed lack of user-facing
  controls over AI-launched processes.
- **Side effects**: New files in `scripts/safety/` and `docs/`. No processes
  affected.

### ~13:xx | Killed orphan python process (PID 3165)
- **Action**: `kill 3165 3163 3161` — killed the parent `uv run` and shell
  wrapper and the actual python process running `run_evals.py --overwrite`.
- **Reason**: Discovered via `ps aux` that a process from 7:44PM the previous
  evening was still running. It had escaped prior `TaskStop` calls because it
  was launched through a `| head -20` pipe — the head closed, the python did
  not. This process had been making API calls for ~7 hours on the old config.
- **Side effects**: OpenRouter spend on Gemini 2.5 Pro and any other models
  the orphan cycled through during its 7-hour life. Exact amount unknown;
  check https://openrouter.ai/activity.

### Earlier in session
- Implemented per-eval temperature support (DAT/CDAT at 1.0/1.5/2.0,
  PACE at 0.0).
- Added seed variation to DAT and CDAT calls (base_seed + trial index) so
  that each call has a unique seed.
- Added explicit top_p and top_k controls to break provider-default nucleus
  filtering that was suppressing diversity.
- Scored Opus 4.5, Opus 4.6, GPT-5.4 on all three metrics and confirmed
  values are within published ranges.

### Notes
- Llama-4-Maverick produces "cloud" as the first DAT word regardless of
  temperature (1.0-2.0), seed, or top_p/top_k. This is a model-level learned
  prior, not a sampling issue.
- Anthropic Claude models appear deterministic at temperature 1.0 via
  OpenRouter (need to verify if this is OpenRouter routing or Anthropic API
  behavior).
