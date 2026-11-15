# Experiments

What was tried to improve answer quality, what happened, and what the system uses today. Most ideas did not help. They are listed so the numbers are on record.

Three notes on reading the numbers:

- Experiments before 11 Nov 2026 were graded by an earlier version of the three-judge panel (judges from several model families). Inside each experiment every arm was graded by the same panel, so the comparison is fair. The absolute scores differ a little from the final table in [evaluation](evaluation.md).
- The noise level is about 0.04 to 0.05 on a score, and the 95% interval on 200 questions is about ±0.08. A difference below that is treated as chance.
- Experiments 1 to 5 and 10 are documented with their final tables only, under `experiments/`. Their scripts and raw files are not in this repository. The scripts for experiments 6 to 9 and 11 are in `eval/results/`.

## Summary

| # | Idea | Result | Used today? |
|---|---|---|---|
| 1 | Give the model the top 3 articles whole (Wix) | Fewer wrong answers on Wix; part of the gain did not hold on the held-out set | Yes, Wix only |
| 2 | Stronger embedding model with the article title in each chunk | Right article found far more often; answers level; 18% fewer tokens | Yes |
| 3 | Larger answer model (mistral-large-4 against ministral-14b) | Wix wrong answers sent 27 to 14, mostly by handing over more cases | Yes |
| 4 | Search with the customer's own words, not a model-written query | Own words found the right TechQA article more often (96 against 91 of 100) | Yes |
| 5 | Rewrite the prompt as reader, writer and checker steps | No gain, twice the tokens | No |
| 6 | Hybrid search (keyword plus dense) | Worse on Wix; no gain on TechQA | No |
| 7 | Cohere reranker | No gain | No |
| 8 | A second model that checks each draft before sending | Net negative | No |
| 9 | Ask the customer a clarifying question when the question is vague | Net negative | No |
| 10 | Whole articles for TechQA | No gain, 2.5 times the tokens | No |
| 11 | More reasoning effort in the answer model (medium, high) | No gain, 1.5 and 2.9 times the median time | No (low) |

## 1. Whole articles for Wix

**Idea.** Wix answers often depend on a fact outside the best-matching chunk. Give the model the top 3 distinct articles whole (12,000 characters each at most) plus the remaining chunks.

**Test.** An offline replay of 200 drafts per arm, then a live copy of the workflow on 100 Wix questions (the tuning set, the questions the change was designed from) and, for the whole-article version only, on the other 100 Wix questions (the held-out set).

**Result.**
- Replay: 7 to 9 more good answers sent and 6 to 8 fewer wrong ones (p = 0.04 and 0.02).
- Live copy, tuning set: good sent 77 to 87, wrong sent 13 to 5 (p = 0.02), at 1.8 times the tokens.
- Held-out set: good 77, wrong 15, handed off 8, score 0.62, against 0.82 on the tuning set. The fix was chosen after reading the failures of the tuning set, so part of that gain was optimistic.

**Decision.** Kept for Wix, where the gain survived on the held-out set in part. Tables: [whole-article tests](experiments/whole-articles.md).

## 2. A stronger embedding model

**Idea.** Replace the previous embedding model with `nvidia/nemotron-3.5-embed-8b`, adding the article title to each chunk and keeping 2,048 dimensions.

**Result** (200 Wix and 100 TechQA questions).
- The right Wix article was in the top 3 for 187 of 200 questions, against 165 before.
- The right document reached the model's context for 94% of questions, against 89%.
- Final answers moved only within noise: 0.71 to 0.74 on a small answer model, 0.75 to 0.75 on mistral-large-4 across 300 questions. Tokens fell 18%.

**Decision.** Adopted: equal answers with fewer tokens, and a simpler path with no reranker call. Tables: [embedding model test](experiments/embedding-model.md).

## 3. A larger answer model

**Test.** 300 offline replays (200 Wix, 100 TechQA), same retrieval, `ministral-14b` against `mistral-large-4`.

**Result.**
- Wix: wrong answers sent 27 to 14 (p about 0.02), good answers sent 167 to 168, hand-offs 6 to 18, score 0.70 to 0.77 (not significant).
- TechQA: wrong sent 11 to 5, but good sent 84 to 77; score level.

The larger model mostly turns wrong answers into cases handed to a person. Its low-confidence signal is informative, and the escalation rule uses it. A first look on 40 questions (0.47 against 0.20) was overstated by how those questions were picked; the 300-question comparison is the one to trust.

**Decision.** Adopted: `mistral-large-4` answers, and its low-confidence signal feeds the escalation rule. Tables: [larger answer model](experiments/larger-model.md).

## 4. The customer's own words as the search query

**Test.** Compare the right document's rank when searching with the question as written against a query written by a model, plus a "hypothetical answer" variant (HyDE).

**Result.** Searching with the customer's own words found the right TechQA document for 96 of 100 questions against 91 for a model-written query. On Wix there was no difference. HyDE added nothing.

**Decision.** The live system embeds the question as written, after the PII step.

## 5. A reader, writer and checker prompt set

**Test.** A three-step prompt set replayed on 40 Wix questions (20 that were wrong, 20 that were good), against the existing prompt re-run.

**Result.** No gain. Of the 20 wrong ones, 4 were fixed by each version. Over all 40: 22 against 23 good and 13 against 15 wrong, at about twice the tokens. The same model repeats the same selection mistakes inside the extra steps, and a checker step built on it cannot see them.

**Decision.** Not used: the single prompt stays. Tables: [reader, writer and checker prompts](experiments/prompt-set.md).

## 6. Hybrid search

**Idea.** Add keyword search. BM25 (k1 1.5, b 0.75) over the chunks, fused with the dense ranking by reciprocal rank fusion (k = 60), top 6 chunks.

**Test.** A first batch of 100 mixed questions (50 Wix and 50 TechQA, picked to include many of the wrong and handed-off cases), then the other 150 Wix questions, so all 200 Wix questions were covered. Same answer model, same judges.

**Result.**
- On the 50 hard Wix questions hybrid looked better (+0.38 against +0.26; a re-run of the baseline gave +0.18).
- On the 150 typical Wix questions it was worse (+0.83 against +0.95). Over all 200 Wix questions: +0.78 to +0.72, 17 good answers lost against 7 gained (p = 0.064).
- Cause: hybrid pushed the right article out of the context on 13 of 200 Wix questions (187 down to 174).
- TechQA (100 questions, hybrid plus rerank): 76 good, 4 wrong, 20 handed off (+0.72) against 77, 7 and 16 (+0.70). No gain.
- Other settings: a smaller fusion constant (k = 2 or 10) still lost the article on 3 questions and changed nearly every ranking. A dense-heavy weighted fusion would keep the article, but the typical questions are already near the ceiling (+0.95), so it was not run.

**Decision.** Not used. Proof: `eval/results/rerank_hybrid/`.

## 7. Cohere reranker

**Test.** Rerank the dense top 30 with `rerank-v3.5` and keep 6, on 50 Wix and 50 TechQA questions. The call takes a median of 0.66 s.

**Result.** No gain: Wix +0.28 against +0.26 for the baseline, and TechQA within ±0.06 of the baseline. Both are inside the noise. Proof: `eval/results/rerank_hybrid/`.

**Decision.** Not used: it adds a call and no measurable gain.

## 8. A second model that checks each draft

**Idea.** Before an answer goes out, a model from another family reads the question, the retrieved passages and the draft, and rates it 1 to 5. Low scores are handed to a person.

**Test.** Laguna rated 343 of the saved answers (all 93 wrong ones and 250 good ones, scaled back up). It did not see the reference.

**Result.** Net negative. Blocking drafts rated below 4 caught 37 of 93 wrong answers but also blocked 86 of 652 good ones (13%), a net loss of 49 points over 745 answers. Rated below 3, it changed almost nothing. 52 of the 93 wrong answers still got a 5. The wrong answers are mostly well supported by the documents, so a reviewer without the reference cannot tell. Proof: `eval/results/extra_step_tests/`.

**Decision.** Not used: it blocks more good answers than wrong ones.

## 9. A clarifying question for vague questions

**Idea.** If the question is too vague, ask the customer for detail instead of answering.

**Test.** Laguna rated how clear each of the 400 live questions is (1 vague, 5 clear), then the live outcomes of the vague ones were checked.

**Result.** Net negative. Of 33 wrong answers, 8 came from vague questions (24%); of 326 good answers, 68 did (21%). Vague questions are answered correctly as often as not, so asking back would replace 68 good answers to avoid 8 wrong ones (about 0.15 lower on the score). Proof: `eval/results/extra_step_tests/clarify_check.py`.

**Decision.** Not used: vague questions are answered correctly about as often as clear ones.

## 10. Whole articles for TechQA

**Test.** The same whole-article change on TechQA, on a live copy.

**Result.** Good sent 76 to 72, wrong sent 8 to 8, escalations 16 to 20, at about 2.5 times the tokens. TechQA technotes are short and the chunks already hold the needed text, so TechQA keeps the 6 chunks.

**Decision.** Not used for TechQA. Tables: [whole-article tests](experiments/whole-articles.md).

## 11. More reasoning effort in the answer model

**Idea.** `mistral-large-4` runs with its reasoning effort set to low. Medium and high let it think longer before answering, which might fix some of the wrong answers.

**Test.** 100 questions: every question that ended wrong or handed off in the live run (Wix 20 and 10, TechQA 13 and 31) plus 13 random good ones per dataset. Each was answered three times through Mistral's own API, with the same retrieved passages and settings, changing only the effort. Wix (43 rows) and TechQA (57 rows) are reported separately. Two judges graded the answers (Nemotron and GLM), and an answer counted as good only if both gave it 4 or more. The third judge is left out of all three efforts, so these scores are not comparable with the final table. A few rows (Wix 2 at medium and 2 at high, TechQA 1 at high) had a judge error and are left out of the counts.

**Result.**

| | Good sent | Wrong sent | Handed off | Score on these rows |
|---|---|---|---|---|
| Wix, low | 16 | 15 | 12 | +0.02 |
| Wix, medium | 16 | 15 | 10 | +0.02 |
| Wix, high | 12 | 18 | 11 | -0.14 |
| TechQA, low | 13 | 11 | 33 | +0.04 |
| TechQA, medium | 14 | 13 | 30 | +0.02 |
| TechQA, high | 14 | 11 | 31 | +0.05 |

Neither medium nor high turned a single wrong answer into a good one on either dataset. The other changes (good to wrong, wrong to hand-off, hand-off to good) are 0 to 2 rows each, inside the noise. Re-running the same low setting on the 20 Wix rows that were wrong in the live run gave 15 wrong, a reminder that rows picked for being wrong move on their own when rerun.

| Effort | Median time | Output tokens | of which reasoning | Total tokens vs low | Cost per 1,000 requests at list price |
|---|---|---|---|---|---|
| low | 7.7 s | 967 | 801 | 1.00x | $3.62 to $7.13 |
| medium | 11.7 s | 1,515 | 1,341 | 1.17x | $4.77 to $9.42 |
| high | 22.2 s | 2,725 | 2,530 | 1.54x | $7.31 to $14.48 |

The input is the same at every effort (2,272 tokens on average). Only the model's reasoning grows.

**Decision.** Not used. The live system keeps low effort: medium and high cost more time and money with no measurable gain on these 100 rows. The sample is small and deliberately hard, so it can rule out a large improvement but not a small one. Proof: `eval/results/effort_test/`.
