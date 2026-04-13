# AI Operations Log

Running log of significant actions I (the AI) take. Every entry should be
specific enough to audit — what I did, when, why, and what state change it
caused.

Format: `YYYY-MM-DD HH:MM | action | reason | side effects`

---

## 2026-04-12

### ~11:20 | Added reasoning control, reordered, relaunched
- **Action**: Killed `bhhnrjv6w` (stuck on QwQ-32B taking 6+ min/call). Added `reasoning.effort=low, exclude=true` support via OpenRouter unified reasoning API. Added retry-without-reasoning fallback for providers that reject the param (QwQ via SiliconFlow). Bumped max_tokens to 1024/2048 to give reasoning models headroom. Reordered: non-reasoning first, reasoning (QwQ, DeepSeek R1, o3-mini, o4-mini, o3) last. Relaunched as `bb1go5ov6` (PID 18280).
- **Result**: QwQ call time dropped from 6+ min to 25s. o3-mini 4.5s. DeepSeek R1 29s.

### ~09:57 | Added async concurrency, relaunched
- **Action**: Killed `baeejhzew`, refactored `run_dat`/`run_cdat`/`run_pace` to async with bounded concurrency (semaphore=20). Relaunched as `bgc0b90xn` (PID 15936).
- **Reason**: Sequential API calls were the bottleneck (~12 min/model for CDAT, ~20 min for PACE). With 20 concurrent calls, PACE for a single model drops to ~30s.
- **Config added**: `concurrency: 20` in yaml; uses OpenRouter async client via `AsyncOpenAI`.
- **Smoke test**: 8 PACE calls for Llama 8B (debug mode) finished in 22s → confirms concurrent dispatch works.

### ~09:45 | Relaunched with per-eval max_tokens caps
- **Action**: Killed task `bek2dptrj`, added `dat_max_tokens=256`, `cdat_max_tokens=256`, `pace_stage1_max_tokens=400`, `pace_stage2_max_tokens=1200` to config. Relaunched as `baeejhzew` (PID 15436).
- **Reason**: User asked about max_tokens limits. Default was 1024. Weak models can hallucinate indefinitely at high temps — caps prevent runaway cost and save time.

### 08:28 | Resumed run after brief pause
- **Action**: Relaunched `src/dat_eval/scripts/run_evals.py configs/dat_eval/run_evals.yaml`.
- **Task ID**: `bek2dptrj` (PID 13234)
- **Reason**: User paused to discuss Llama 8B's high failure rate at temp=1.5 (16/40 trials yielded <7 valid words). Decision: keep current config and filter/adjust post-hoc during scoring.
- **State at resume**: llama-3.1-8b-instruct has DAT at temp 1.0 (40/40 valid) and temp 1.5 (24/40 valid) complete. Run continues with temp 2.0 for it, then next model.

### 04:28 | Launched full DAT/CDAT/PACE run with budget cap
- **Action**: Started `src/dat_eval/scripts/run_evals.py configs/dat_eval/run_evals.yaml` in background.
- **Task ID**: `bdovfrkzu` (killable via TaskStop or `bash scripts/safety/kill_all.sh`)
- **PID**: 11506
- **Configuration**:
  - 49 models in cheap-to-expensive order
  - Budget cap: $30.00 (remaining OpenRouter credit is ~$34)
  - DAT: temps 1.0, 1.5, 2.0 × 40 trials × top_p=1.0, top_k=0, unique seed per trial
  - CDAT: same temps × 50 cues × top_p=1.0, top_k=0, unique seed per cue
  - PACE: temp=0.0 × 50 seeds (6 models already have PACE complete)
- **Expected**: ~42 models will run to completion; last 7 (Claude Sonnet 4, gpt-5, gpt-5.4, gpt-4-turbo, Opus 4.5/4.6, o3) will be skipped by budget cap.
- **Estimated spend**: ~$28 total for this run.
- **User approval**: explicit "go" in chat.

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
