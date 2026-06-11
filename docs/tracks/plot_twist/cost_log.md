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
