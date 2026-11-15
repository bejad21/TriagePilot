# Live evaluation

200 Wix and 200 TechQA questions sent through the running webhooks, with the answers and judge grades saved. This is the run behind the headline numbers in the [README](../../../README.md) and [docs/evaluation.md](../../../docs/evaluation.md). Wix and TechQA are two separate datasets and are never pooled.

## What was running

Webhook, input checks, embed the question (`nvidia/nemotron-3.5-embed-8b`, query side, first 2,048 dimensions), search Qdrant (top 6 chunks, dense only), then one call to `mistral-large-4` (reasoning effort low, temperature 0), parse the reply, outbound check and escalation rule. For Wix the top 3 articles are read whole (12,000 characters each at most, 60,000 in total). For TechQA the answer model gets the top 6 chunks. Both input and output checks use `nvidia/nemotron-3.5-bolt-e4b`.

## Result

Good = at least 2 of the 3 judges (Nemotron-3.5-super, GLM-4.5-flash, Laguna-xs-2.1) give correctness 4 or more. Score = (good sent - wrong sent) / questions, and a hand-off counts as zero. [`final_numbers`](../final_numbers/README.md) regenerates the table from the saved files.

| | Wix | TechQA |
|---|---|---|
| Answered / handed to a person | 190 / 10 | 169 / 31 |
| Good answers sent | 170 | 156 |
| Wrong answers sent | 20 (10.5% of sent) | 13 (7.7% of sent) |
| Score (95% interval) | +0.750 (+0.660 to +0.835) | +0.715 (+0.635 to +0.795) |

## Files

| File | Purpose |
|---|---|
| `live_eval.py` | Sends the questions to the webhooks (`run`), grades the answers (`judge`) and reports (`report`). Resumable. |
| `live_laguna.py` | Grades the answered rows with Laguna and prints the panel result and judge agreement. |
| `guard_inputs.json` | The 230 labelled guard inputs: 42 attacks, 38 look-alikes, 150 real questions. |
| `guard_probe.py` | Sends the 42 labelled attacks to a live webhook (see [security tests](../../../docs/security-tests.md)). |
| `guard_offline.py` | Runs all 230 inputs through the guard model with the workflow's own prompt and counts the verdicts. |
| `smoke.py` | Sends five test questions to a webhook (an answerable one, off-topic, an injection, empty text, one with an email address). |
| `work/` | Saved answers (`live_wix.jsonl`, `live_techqa.jsonl`) and judge grades (`judge_*.jsonl`). |

Run the scripts from `eval/` with the keys loaded. Each docstring has its usage line.
