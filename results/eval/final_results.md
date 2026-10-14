# Pairwise evaluation (final)

Judge: `google/gemini-3.6-flash` via kaggle_benchmarks, temperature 0, 5 pairs/request, >=5 s between requests, 3 attempts per batch. Every pair judged in both A/B orders; per-pair scores summed (win=1, tie=0.5, loss=0), so a verdict that flips with order becomes a tie. Win rate excludes ties; CI is Wilson 95%; p is a two-sided sign test.

| Comparison (candidate vs baseline) | Eval set | n | W | T | L | Win rate (excl. ties) | 95% CI | Ties = ½ | p | Order-consistent | Judge picked A | Mean tokens cand/base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b-dpo-uf vs qwen2.5-0.5b-sft | alpaca | 300 | 111 | 78 | 111 | 0.500 | 0.435–0.565 | 0.500 | 1.000 | 0.74 | 0.60 | 247 / 184 |
| qwen2.5-0.5b-dpo-hh vs qwen2.5-0.5b-sft | hh | 300 | 144 | 80 | 76 | 0.654 | 0.590–0.714 | 0.613 | <0.001 | 0.75 | 0.60 | 136 / 170 |
| smollm2-360m-lora-dpo vs smollm2-360m-instruct | persona | 24 | 8 | 4 | 12 | 0.400 | 0.219–0.613 | 0.417 | 0.503 | 0.58 | 0.62 | 215 / 256 |
| smollm2-360m-lora-dpo-lr1e-4 vs smollm2-360m-instruct | persona | 24 | 7 | 1 | 16 | 0.304 | 0.156–0.509 | 0.312 | 0.093 | 0.96 | 0.48 | 128 / 256 |
| qwen2.5-0.5b-dpo-hh vs qwen2.5-0.5b-sft | alpaca | 300 | 98 | 107 | 95 | 0.508 | 0.438–0.578 | 0.505 | 0.886 | 0.72 | 0.62 | 161 / 184 |
| qwen2.5-0.5b-dpo-uf vs qwen2.5-0.5b-sft | hh | 300 | 95 | 75 | 130 | 0.422 | 0.359–0.487 | 0.442 | 0.023 | 0.74 | 0.59 | 241 / 170 |
| qwen2.5-0.5b-sft vs qwen2.5-0.5b-base | alpaca | 300 | 134 | 58 | 108 | 0.554 | 0.491–0.615 | 0.543 | 0.108 | 0.82 | 0.59 | 184 / 233 |

## Single-pass results (each order separately)

| Comparison | Pass | W/T/L | Win rate (excl. ties) |
|---|---|---|---|
| dpo_uf_vs_sft__alpaca | original order | 150/5/145 | 0.508 |
| dpo_uf_vs_sft__alpaca | swapped order | 145/7/148 | 0.495 |
| dpo_hh_vs_sft__hh | original order | 183/12/105 | 0.635 |
| dpo_hh_vs_sft__hh | swapped order | 169/15/116 | 0.593 |
| lora_dpo_vs_instruct__persona | original order | 6/3/15 | 0.286 |
| lora_dpo_vs_instruct__persona | swapped order | 8/3/13 | 0.381 |
| lora_dpo_lr1e-4_vs_instruct__persona | original order | 7/0/17 | 0.292 |
| lora_dpo_lr1e-4_vs_instruct__persona | swapped order | 8/0/16 | 0.333 |
| dpo_hh_vs_sft__alpaca | original order | 136/33/131 | 0.509 |
| dpo_hh_vs_sft__alpaca | swapped order | 136/32/132 | 0.507 |
| dpo_uf_vs_sft__hh | original order | 125/5/170 | 0.424 |
| dpo_uf_vs_sft__hh | swapped order | 136/4/160 | 0.460 |
| sft_vs_base__alpaca | original order | 156/7/137 | 0.532 |
| sft_vs_base__alpaca | swapped order | 164/6/130 | 0.558 |

## Persona LoRA-DPO vs SmolLM2-360M-Instruct (combined over both orders)

| Model | Criterion | W | T | L | Win rate (excl. ties) | Order-consistent |
|---|---|---|---|---|---|---|
| smollm2-360m-lora-dpo | style | 4 | 20 | 0 | 1.000 | 0.92 |
| smollm2-360m-lora-dpo | usefulness | 8 | 4 | 12 | 0.400 | 0.62 |
| smollm2-360m-lora-dpo | safety | 1 | 21 | 2 | 0.333 | 0.83 |
| smollm2-360m-lora-dpo | overall | 8 | 4 | 12 | 0.400 | 0.58 |
| smollm2-360m-lora-dpo-lr1e-4 | style | 23 | 1 | 0 | 1.000 | 0.92 |
| smollm2-360m-lora-dpo-lr1e-4 | usefulness | 6 | 1 | 17 | 0.261 | 0.96 |
| smollm2-360m-lora-dpo-lr1e-4 | safety | 4 | 16 | 4 | 0.500 | 0.92 |
| smollm2-360m-lora-dpo-lr1e-4 | overall | 7 | 1 | 16 | 0.304 | 0.96 |

## Per category: qwen2.5-0.5b-dpo-hh vs qwen2.5-0.5b-sft (HH eval)

| Category | W | T | L | Win rate (excl. ties) |
|---|---|---|---|---|
| benign_helpfulness | 15 | 4 | 11 | 0.577 |
| calibration_and_uncertainty | 12 | 9 | 9 | 0.571 |
| cyber_abuse | 13 | 9 | 8 | 0.619 |
| harmful_instructions | 14 | 9 | 7 | 0.667 |
| hate_or_harassment | 17 | 8 | 5 | 0.773 |
| legal_financial_safety | 17 | 5 | 8 | 0.680 |
| medical_safety | 14 | 7 | 9 | 0.609 |
| privacy_and_doxxing | 20 | 6 | 4 | 0.833 |
| safe_transformation | 13 | 10 | 7 | 0.650 |
| self_harm_support | 9 | 13 | 8 | 0.529 |

## Per category: qwen2.5-0.5b-dpo-uf vs qwen2.5-0.5b-sft (HH eval)

| Category | W | T | L | Win rate (excl. ties) |
|---|---|---|---|---|
| benign_helpfulness | 15 | 4 | 11 | 0.577 |
| calibration_and_uncertainty | 5 | 12 | 13 | 0.278 |
| cyber_abuse | 6 | 9 | 15 | 0.286 |
| harmful_instructions | 7 | 11 | 12 | 0.368 |
| hate_or_harassment | 12 | 3 | 15 | 0.444 |
| legal_financial_safety | 13 | 7 | 10 | 0.565 |
| medical_safety | 7 | 6 | 17 | 0.292 |
| privacy_and_doxxing | 9 | 13 | 8 | 0.529 |
| safe_transformation | 13 | 5 | 12 | 0.520 |
| self_harm_support | 8 | 5 | 17 | 0.320 |
