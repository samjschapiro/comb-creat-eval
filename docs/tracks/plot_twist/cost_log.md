# plot_twist OpenRouter cost ledger

Ground-truth cumulative spend on the API key (OpenRouter `/key` usage), logged per run.

| timestamp | run | cumulative ($) | Δ this run ($) |
|---|---|---|---|
| 2026-06-11 14:38 UTC | baseline (8-model gen+score+annotate, AGC tier gen done, AGC scoring in progress) | $4.2381 | $4.2381 |
| 2026-06-11 14:41 UTC | AGC tier scoring (663 stories x 3 judges) | $9.0821 | $4.8440 |
| 2026-06-11 14:43 UTC | AGC tier annotation | $9.2679 | $0.1858 |
| 2026-06-11 15:12 UTC | AGC tier-2 generation (12 models x 30) | $17.8768 | $8.6089 |
| 2026-06-11 15:16 UTC | AGC tier-2 scoring | $25.4776 | $7.6008 |
| 2026-06-11 15:18 UTC | AGC tier-2 annotation | $26.3701 | $0.8925 |
| 2026-06-11 15:35 UTC | twist mechanism classification | $26.4693 | $0.0992 |
| 2026-06-11 16:43 UTC | opus-4.8 + gpt-5.5 generation+scoring | $35.1126 | $8.6433 |
| 2026-06-11 16:44 UTC | opus-4.8 + gpt-5.5 annotation | $35.5585 | $0.4459 |
| 2026-06-11 18:42 UTC | AGC full sweep (36 models) gen+scoring | $72.1563 | $36.5978 |
| 2026-06-11 18:45 UTC | AGC full sweep annotation | $73.5170 | $1.3607 |
| 2026-06-11 20:19 UTC | realism scoring (gemini-flash) | $75.5432 | $2.0262 |
| 2026-06-11 20:33 UTC | realism re-score (sonnet-4, tightened anchors) | $96.0342 | $20.4910 |
| 2026-06-12 18:06 UTC | PRE thinking smoke (baseline) | $101.7997 | $5.7655 |
| 2026-06-12 18:14 UTC | thinking smoke (11 models x 2 levels, max_tokens=16000) | $102.9564 | $1.1567 |
| 2026-06-12 18:19 UTC | PRE Exp1 thinking intervention (9 models x 3 levels x 8, 32k) | $102.9686 | $0.0122 |
| 2026-06-12 18:26 UTC | Exp3 structural extraction (98 stories x 2 extractors) | $110.2814 | $7.3128 |
| 2026-06-12 19:51 UTC | PRE Exp1 rubric scoring (202 stories x 3 judges) | $118.1325 | $7.8511 |
| 2026-06-12 20:00 UTC | PRE Exp1 realism scoring (202 stories, sonnet-4) | $123.1396 | $5.0071 |
| 2026-06-12 20:01 UTC | PRE Exp1 full analysis (annotate reveals) | $125.6067 | $2.4671 |
| 2026-06-17 | realism 3-judge ensemble (add gpt-4o + gemini-flash to 2070 stories; claude reused from cache) | $142.48 | $16.76 |
| 2026-06-20 (reconcile) | unlogged spend between 2026-06-17 and Exp2 (live /key reconciliation) | $145.1715 | $2.6915 |
| 2026-06-20 | Exp2 prompt-methods smoke (2 models x 2 methods x 2) | $145.8984 | $0.7269 |
| 2026-06-20 | Exp2 prompt-methods generation (5 models x 2 methods x 8 = 80 stories + summaries; baseline reused) | (incl. below) | — |
| 2026-06-20 | Exp2 prompt-methods rubric (80 x 3) + realism (80) + analysis (annotate 120) | $151.8201 | $5.9217 |
| 2026-06-20 | Exp2 expand 5->8 models (add sonnet-4.6, nemotron-3-super-120b, cogito-v2.1-671b; +retries; rescore new ~42) | $154.0916 | $2.2715 |
