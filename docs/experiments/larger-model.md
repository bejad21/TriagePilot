# Larger answer model

Experiment 3 in [experiments](../experiments.md). The question: does a larger answer model help? It sends fewer wrong answers on Wix, mostly by handing more cases to a person, and it is level on TechQA. This page holds the final tables from the saved run outputs. The scripts and raw files of this run are not part of this repository.

## Which models answered in usable time

One real prompt (about 3,500 tokens) was sent to 12 hosted models, then again with reasoning turned down. Only two answered in usable time:

- `nvidia/nemotron-3.5-lightning-30b-a3b`, in about 4 s.
- `mistralai/mistral-large-4-1t-a52b` with `reasoning_effort: low`, in about 24 s alone.

Gemma-4 31B, DeepSeek-V4.1-flash, gpt-oss-20b, Step-5 and MiniMax-M3.1-flash timed out after 120 to 150 s.

## Full comparison

300 questions (Wix 200, TechQA 100), the same saved passages, the same prompt and the same judges. `ministral-14b` against `mistral-large-4`. An answer counts as sent when it is answered, confidence is not low and `needs_human_review` is not true. The panel is the majority of three judges.

| | Model | Good sent | Wrong sent | Handed off | Accuracy when answering | Score (good - wrong) / N |
|---|---|---|---|---|---|---|
| Wix, tuning set | ministral-14b | 88 | 9 | 3 | 91% | 0.79 |
| | mistral-large-4 | 90 | 3 | 7 | 97% | 0.87 |
| Wix, held-out set | ministral-14b | 79 | 18 | 3 | 81% | 0.61 |
| | mistral-large-4 | 78 | 11 | 11 | 88% | 0.67 |
| **Wix, all 200** | ministral-14b | 167 | 27 | 6 | 86% | 0.70 |
| | mistral-large-4 | 168 | 14 | 18 | 92% | 0.77 (change +0.07, range -0.02 to +0.16) |
| TechQA, 100 | ministral-14b | 84 | 11 | 5 | 88% | 0.73 |
| | mistral-large-4 | 77 | 5 | 18 | 94% | 0.72 (change -0.01, range -0.10 to +0.08) |

- Wix wrong sent: 20 removed and 7 added (sign test p about 0.02). Good sent: +11 and -10.
- TechQA wrong sent: 7 removed and 1 added (p about 0.07). Good sent: +1 and -8 (p = 0.039, worse).
- The large model's hand-offs are well chosen. Of its 18 Wix hand-offs only 4 had a good draft, and of its 18 TechQA hand-offs only 1.
- Tokens per request rose about 10% (Wix 4,670 against 4,220, TechQA 2,930 against 2,690).
- A 40-question probe (score 0.47 against 0.20) overstated the gain, because its 20 wrong questions were chosen for being wrong with the smaller model. The 300-question comparison is the one to use.

## Limits

One-shot replay on saved passages (no search step, no hand-off rule beyond confidence), AI judges, 100 to 200 questions.
