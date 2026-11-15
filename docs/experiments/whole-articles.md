# Whole-article tests

Experiments 1 and 10 in [experiments](../experiments.md). The question: should the answer model get the top 3 articles whole instead of only the best chunks? It helps on Wix and does not on TechQA. This page holds the final tables from the saved run outputs. The scripts and raw files of these runs are not part of this repository.

## Why wrong answers were sent

Run of 200 saved drafts graded by a three-judge panel (good = correctness 4 or 5): 20 wrong answers sent on Wix and 10 on TechQA.

| Cause | Wix | TechQA |
|---|---|---|
| Right article retrieved, but the chunk with the needed fact was not among the 6 the model saw | 9 | 2 |
| Wrong or look-alike article retrieved, or the source article missed | 2 | 5 |
| The fact was in the passages and the model misused it (including agreeing with the customer's false premise) | 2 | 2 |
| Reference needs facts that are not in its article or not in the corpus | 5 | 0 |
| Borderline, judges split | 2 | 1 |

The median article is 2.6k characters on Wix and 2.8k on TechQA, and 94% and 88% are 8k or shorter, so giving the model whole articles is feasible.

## Replay: chunks (A) against whole articles (F)

Same prompt, same output schema, one shot, 200 drafts per arm, 100 questions per set. Two different panels of three judges (panel 1 and panel 2). Paired bootstrap and exact McNemar test on good answers sent.

| Set | Panel | Good sent, A to F | Wrong sent, A to F | Change in good sent (95% range) | McNemar p |
|---|---|---|---|---|---|
| Wix | 1 | 82 to 89 | 15 to 9 | +7 (+2 to +13) | 0.039 |
| Wix | 2 | 76 to 85 | 21 to 13 | +9 (+2 to +16) | 0.022 |
| TechQA | 1 | 78 to 81 | 15 to 10 | +3 (-2 to +8) | 0.453 |
| TechQA | 2 | 79 to 80 | 14 to 11 | +1 (-5 to +7) | 1.000 |

Mean prompt tokens, A to F: Wix 1,680 to 3,686, TechQA 2,412 to 5,827.

## Live copies, 100 questions of each dataset (the Wix tuning set)

The change was run as separate copies of the workflows, with the same questions sent one at a time.

| | Version | Sent | Good sent | Wrong sent | Sent to a person | Tokens per question (estimate) |
|---|---|---|---|---|---|---|
| Wix | chunks | 90 | 77 | 13 | 10 | about 3,800 |
| Wix | whole articles | 92 | **87** | **5** | 8 | 6,869 |
| TechQA | chunks | 84 | 76 | 8 | 16 | about 6,100 |
| TechQA | whole articles | 80 | 72 | 8 | 20 | 15,120 |

- Wix: good sent +10 (95% range +3 to +18, exact McNemar p = 0.021), wrong sent -8 (range -15 to -2), at about 1.8 times the tokens.
- TechQA: good sent -4 (range -12 to +4, p = 0.454), wrong sent unchanged, more hand-offs, at about 2.5 times the tokens.

## Held-out Wix questions

The other 100 Wix questions (the held-out set), whole-article version only. The tuning set, the 100 questions above, was used to design the change.

| Wix, whole articles | Tuning set | Held-out set | Both (200) |
|---|---|---|---|
| Good sent | 87 | 77 | 164 |
| Wrong sent | 5 | 15 | 20 |
| Handed off | 8 | 8 | 16 |
| Accuracy when it answers | 95% | 84% | 89% |
| Score, (good - wrong) / questions | 0.82 (0.72 to 0.91) | 0.62 (0.47 to 0.76) | 0.72 |

Wrong sent 5 against 15 gives Fisher exact p = 0.032, so the tuning-set result was optimistic. Of the 15 wrong answers, the source article was given whole in 9, missing from the search in 5 and partly seen in 1.

## Limits

- The baseline is an earlier saved run, not a same-day control. Run-to-run noise is about 5 per 100.
- The change was chosen after reading failures on the same questions.
- One run each, and AI judges disagree on several answers.
- The TechQA result is inside the noise for the replay. Likely reasons it did not improve on the live copy, not tested: TechQA articles are longer and more alike, and the model hedges more.
