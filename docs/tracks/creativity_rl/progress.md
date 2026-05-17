# creativity_rl — MCNS-style RL for intrinsic novelty

## Research question

Does Minimal Criteria Novelty Search applied to LM preference optimization (MCNS-RL) produce models that are measurably more creative on held-out prompts than vanilla RLHF, *without* explicit quality optimization?

Strong form (Lehman & Stanley 2010 headline test): does MCNS-RL match or beat a vanilla-RLHF baseline on judge-rated *quality* on held-out prompts, despite never optimizing for quality?

See [docs/memos/mcns_dpo.md](../../memos/mcns_dpo.md) for the formal setup. This track implements the M2 (direct-reward RL) variant.

## Reward

$$
r_\text{MCNS}(y, x) \;=\; N(y, x; \mathcal{A}_t) \cdot \mathbb{1}[A(y, x) > \tau].
$$

- $A(y, x)$: scalar from an off-the-shelf RM head (appropriateness gate).
- $N(y, x; \mathcal{A}_t)$: mean cosine distance from $\varphi(y, x)$ to its k nearest neighbors in a growing per-prompt-cluster archive of past responses that passed the gate.
- $\varphi$: SBERT `all-mpnet-base-v2` embedding of $y$ (direct CDAT compatibility).
- $\tau$: calibrated so ~80% of $\pi_\text{ref}$ samples pass; re-calibrated periodically.

## Decisions locked

| Question | Choice | Rationale |
|---|---|---|
| Dataset (train) | `liweijiang/infinite-chats-taxonomy` (~26K open-ended) | Open-ended only; full set, sample for training. |
| Dataset (test) | `liweijiang/infinite-chats-eval` (100 prompts) | Curated by Hivemind authors; never seen in training. Strict OOD. |
| Prompt filtering | Pre-filtered open-ended (no code/math) | Done by Hivemind authors. |
| Appropriateness RM | Off-the-shelf (start: Skywork-Reward-Llama-3.1-8B or ArmoRM "helpfulness" head) | Faster start; train custom classifier later if RM hacking observed. |
| Archive scope | Per-prompt-cluster | Compromise between per-prompt sparsity and global semantic conflation. |
| Behavior char. $\varphi$ | SBERT `all-mpnet-base-v2` | Matches CDAT eval metric exactly. |
| Distance | Cosine | Standard for SBERT. |
| Archive index | FAISS HNSW | Standard kNN; bounded admission via novelty threshold $\rho$. |
| RL algorithm | GRPO | No value head; group-of-K naturally matches "compute novelty within group." |
| Group size $K$ | 8 (initial) | Standard GRPO default; tune. |
| Base models | Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct (start) | Fast iteration. Scale to 3B / 7B for headline runs. |
| Adapter | LoRA, rank 16, $\alpha=32$, attn + MLP | Standard. |
| KL coefficient $\beta$ | 0.05 (initial) | Standard RLHF default; watch KL during training. |
| Logging | wandb | Project: `comb-creat-eval`, group: `creativity_rl`. |

## Pipeline

```
Step 0 (one-time):
  download datasets       → data/creativity_rl/datasets/
  cluster training prompts → data/creativity_rl/prompt_clusters.json
  calibrate τ              → data/creativity_rl/calibration/

Step 1 (training):
  train_grpo.py            → data/creativity_rl/runs/<run_id>/
    on-policy sample K per prompt
    score appropriateness  (frozen RM)
    embed via SBERT
    novelty kNN against per-cluster archive
    r_MCNS = N · 1[A > τ]
    GRPO update with KL to π_ref
    admit y to archive if A > τ and N > ρ
    log to wandb

Step 2 (evaluation):
  eval.py                  → data/creativity_rl/runs/<run_id>/downstream/eval/
    on infinite-chats-eval (100 prompts):
      held-out novelty (SBERT)
      held-out novelty (BGE-large, sanity-check different φ)
      appropriateness retention (% pass τ under π_θ vs π_ref)
      diversity within-prompt (K=10 samples per prompt)
    on dat_eval prompts:
      DAT, CDAT, PACE
    judge-rated quality (Stanley headline test):
      pairwise judge on 100 held-out prompts vs baselines
```

## Baselines (all required)

1. **Base model** (no fine-tune).
2. **SFT** on same training prompts with vanilla teacher responses.
3. **Vanilla GRPO** with appropriateness reward only (no novelty term). Same RM, same KL, same compute. Isolates novelty contribution.

Without (3), we cannot separate "MCNS works" from "any RL on this RM works."

## Open decisions

- Embedding source for prompt clustering and $\varphi$: SBERT for both, or different encoders? (Default: SBERT for both, simpler.)
- Number of prompt clusters: 50? 200? Scales archive granularity.
- Archive admission threshold $\rho$: novelty percentile cutoff. Start: median of in-batch novelties.
- Archive eviction policy when bounded: random or oldest-first.
- Whether to anneal $\tau$ upward (curriculum) as $\pi_\theta$ improves.
- Multi-turn or single-turn only. Default: single-turn for now.

## Risks (from design discussion)

1. **Embedding-novelty gaming.** Mitigation: report novelty under a second embedding at eval time.
2. **RM exploitation.** Mitigation: manual spot-check 50–100 generations per iteration.
3. **Archive saturation.** Mitigation: novelty-thresholded admission.
4. **Reward sparsity early in training.** Mitigation: anneal $\tau$ upward from a low start.
5. **GRPO instability near $\tau$ (hard gate).** Mitigation: soft-gate fallback ($\sigma((A-\tau)/T)$, small $T$).

## Progress

### 2026-05-15 — Track scaffolded + first smoke test

- [x] Track structure created (`src/creativity_rl/`, `configs/creativity_rl/`, `scripts/creativity_rl/`, `data/creativity_rl/`).
- [x] Initial configs: `llama_1b_mcns.yaml` (full run) + `smoke_test.yaml` (50 steps, Qwen-1.5B + OpenAssistant DeBERTa RM, public/ungated models).
- [x] Implementation: `archive.py`, `novelty.py`, `reward.py`, `data.py`, `scoring.py`, `reward_callable.py`, `scripts/train_grpo.py`.
- [x] WandB + Lambda + HF credentials in `.env` (gitignored).
- [x] Lambda A100 (40GB SXM4, us-west-2) provisioned; repo rsynced; `uv sync --extra creativity_rl` succeeded with torch 2.11 / TRL 0.17 / peft 0.19 / faiss-cpu / bitsandbytes.
- [x] Smoke test: 3 iterations debugging archive bootstrap; final smoke (50 steps) ran clean with archive growing 8→15 (≈4% post-warmup admission rate). Wandb: [smoke_test_v1](https://wandb.ai/schapirolab/comb-creat-eval/runs/rxk1abui).
- [x] $\tau$ calibration script (`scripts/calibrate_tau.py`); calibrated $\tau = -2.594$ for Qwen-1.5B + DeBERTa-OA RM (300 base-policy samples, target pass-rate 0.80).
- [ ] Full run v1: MCNS-RL on Qwen-1.5B with 2K prompts, 1500 steps (in flight, launched 2026-05-15 16:21 UTC).
- [ ] Step 2: baseline runs (base, SFT, vanilla-GRPO on appropriateness only).
- [ ] Step 3: held-out eval + Stanley headline test.

#### Archive bootstrap fix (iterated three times during smoke)

| Variant | Archive end (50 steps) | Issue |
|---|---:|---|
| No bootstrap protection | 11 | Empty-archive's artificial novelty=1.0 was recorded, pinning median rho near 1.0 forever — only entries with rare N>1 got in. |
| Warmup + record warmup novelties | 8 | Same problem one level deeper: warmup-period novelty values are artificially high (small archive). Median still inflated. |
| Warmup + skip-record during warmup | 15 | Median rho built only from post-warmup novelties — realistic. Continued admission post-warmup at ~4%. |

#### Smoke-test simplifications (vs full run)

- Train data = `infinite-chats-eval` (50 prompts) instead of `infinite-chats-taxonomy` — smallest dataset, fastest iteration.
- Single global cluster (`n_clusters=1`) — skips KMeans, archive is one global HNSW.
- No $\tau$ calibration (`τ = -2.0` placeholder for DeBERTa RM).
- Qwen2.5-1.5B-Instruct + OpenAssistant DeBERTa RM (public/ungated). Full runs will use Skywork-Reward-Llama and Llama/Qwen base.
- K = 4 generations, 128 max_new_tokens, 50 steps.

#### Files added this session

| File | Purpose |
|---|---|
| `src/creativity_rl/data.py` | HF dataset loading + KMeans prompt clustering. |
| `src/creativity_rl/scoring.py` | `AppropriatenessScorer` (handles chat-template + pair RMs), `SBERTEmbedder`. |
| `src/creativity_rl/reward_callable.py` | TRL-compatible `MCNSRewardFunction`; routes responses to cluster archives, applies MCNS reward, admits qualifying responses to archive, exposes telemetry for wandb. |
| `configs/creativity_rl/smoke_test.yaml` | Smoke-test config. |
| `scripts/creativity_rl/run_smoke.sh` | Smoke-test bash wrapper. |

Updated:
- `src/creativity_rl/scripts/train_grpo.py` — full implementation (7-step pipeline with TRL GRPOTrainer wiring).
- `pyproject.toml` — `creativity_rl` optional extras (torch, trl, peft, bitsandbytes, faiss-cpu, scikit-learn, accelerate, wandb).
- `.env.example` — added LAMBDA_CLOUD_API_KEY, LAMBDA_SSH_KEY_PATH, HF_TOKEN placeholders.

### Next session

1. Download `infinite-chats-taxonomy` and `infinite-chats-eval`; verify open-ended filter.
2. Implement prompt clustering script (KMeans on SBERT embeddings, k≈100).
3. Implement $\tau$ calibration script (sample 500 responses from base model, score with RM, set $\tau$ at 20th percentile).
4. Wire up GRPO loop with TRL; smoke test.
