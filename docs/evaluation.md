# Evaluation

**Summary.** 200 questions were sent through each live webhook. The score is +0.750 on Wix and +0.715 on TechQA (good answers sent minus wrong answers sent, divided by all questions; a hand-off counts as zero). Three judge models from different families graded each answer against an expert reference. The main limit is that a correct answer that differs from the reference counts as wrong.

Wix and TechQA are two independent deployments, each with its own knowledge base, workflow and webhook, and each is evaluated on its own. The numbers are never pooled across the two. Everything below was measured on the running system: 200 questions were sent to each live webhook, and the replies were graded afterwards. Experiments that used offline replays are marked as such in [experiments](experiments.md).

## What was measured

| | Wix | TechQA |
|---|---|---|
| Knowledge base | 1,200 help-center articles | 496 IBM technotes |
| Questions | 200 customer questions with expert-written reference answers | 200 forum questions, each tied to one technote |
| Source | WixQA (`Wix/WixQA`) | TechQA-RAG-Eval (`nvidia/TechQA-RAG-Eval`) |

Each question was sent once to its webhook. All 400 requests ended with a valid response (a failed request was retried). The outcome of a request is one of:

- **answered**: the system sent an answer;
- **escalated**: the system handed the case to a person (the search was empty, the model reported low confidence, or it asked for review).

Answered replies were then graded against the reference answer by a panel of three judges.

## The score

- **Good answer sent**: at least 2 of the 3 judges rate it 4 or higher out of 5.
- **Wrong answer sent**: answered, but not good.
- **Handed off**: escalated. A hand-off counts as zero: it is neither a win nor a loss.
- **Score** = (good sent minus wrong sent) / all questions.

This penalizes a confident wrong answer as much as it rewards a correct one, and it does not reward declining to answer. The 95% intervals come from resampling the questions 5,000 times.

## Results

Judge panel: Nemotron-3.5-super (NVIDIA), GLM-4.5-flash (Z.ai) and Laguna-xs-2.1 (Poolside).

| | Wix (200) | TechQA (200) |
|---|---|---|
| Answered / escalated | 190 / 10 | 169 / 31 |
| Good answers sent | 170 | 156 |
| Wrong answers sent | 20 (10.5% of answers sent) | 13 (7.7% of answers sent) |
| **Score** | **+0.750** (95% interval +0.660 to +0.835) | **+0.715** (+0.635 to +0.795) |
| Latency per request | median 161 s, 90th percentile 413 s, longest 1,468 s | median 190 s, 90th percentile 509 s, longest 1,737 s |

Latency is high because the answer model runs on a shared free endpoint. The same call replayed offline had a median of 37 to 42 s, and a request trace shows nearly all the time in that one step ([operations](operations.md)).

Eight questions have an unclear reference answer (listed with reasons in [unclear labels](../eval/results/final_numbers/unclear_labels.md)). They stay in the numbers above. With them left out the scores are +0.786 on Wix and +0.750 on TechQA.

### How much two runs differ

The score is noisy. Replaying the same system offline a second time on the same 100 questions changed the outcome of 7 of them and moved the score by about 0.04 (0.08 on the Wix half, 0.00 on the TechQA half). So differences smaller than about 0.05 are treated as chance in this project, and the intervals above are about ±0.08 wide.

### What the remaining 33 wrong answers are

Each wrong answer was read next to its reference and the documents.

| Cause | Wix | TechQA |
|---|---|---|
| Reference does not fit the question well (unclear label) | 4 | 4 |
| The right document never reached the model | 4 | 2 |
| The right material was there, but the answer is partial or differs from the expected fix | 8 | 7 |
| The model followed a similar but wrong page | 4 | 0 |

Two examples. In Wix 197 the reference describes Editor X, while the customer uses the normal Wix Editor and the system's answer fits the Wix Editor. In TechQA 36 the reference ("contact your vendor") only fits a Student Version buyer, and the question never says that. This breakdown is my own reading of each case, not an independent review.

## Retrieval and answer-quality metrics

The correctness score says whether an answer matches the reference. These metrics look at the parts of the pipeline separately: did the search find the right material, did the answer stay inside it, and did it address the question. They follow the RAGAS definitions and are reported for each dataset on its own.

| Metric | What it measures | Wix | TechQA |
|---|---|---|---|
| Faithfulness | Share of the answer's claims that the passages the model saw support. The answer is split into single claims and a judge checks each against the passages. A low value means invented content. Mean of two judges (Nemotron and Laguna). | 0.98 (0.98 to 0.99) | 0.95 (0.93 to 0.96) |
| Answer relevancy | Whether the answer addresses the question. A judge writes up to 3 questions the answer would respond to, and the score is their cosine similarity to the real question. The absolute value depends on the embedding model, so it is useful for comparing runs, not as a pass mark. | 0.62 (0.60 to 0.63) | 0.58 (0.56 to 0.60) |
| Context precision | How many of the 6 retrieved chunks are relevant to the reference answer, weighted so that relevant chunks near the top score higher. | 0.89 (0.85 to 0.92) | 0.88 (0.84 to 0.91) |
| Context recall | Share of the reference answer's statements that can be found in the 6 retrieved chunks. | 0.84 (0.80 to 0.87) | 0.87 (0.83 to 0.91) |
| Hit rate at 3 articles | A known source document is among the first 3 distinct articles the search returns. | 0.94 (0.91 to 0.97) | 0.94 (0.89 to 0.98) |
| Hit rate at 6 chunks | A known source document is among the 6 retrieved chunks. | 0.94 (0.91 to 0.97) | 0.95 (0.90 to 0.99) |
| Mean reciprocal rank | 1 divided by the rank of the first source document among distinct articles; 1.0 means it is always first. | 0.82 (0.78 to 0.87) | 0.92 (0.86 to 0.97) |

Intervals are 95%, from resampling the rows. Faithfulness and answer relevancy cover the answers that were sent (190 on Wix, 169 on TechQA). The other metrics cover all questions, because the search runs before the decision to answer.

**How these were measured.** The webhook returns the answer but not the passages it used. The passages come from running the same search the live workflow runs, against the live Qdrant collections (same embedding model, same top 6 chunks, and for Wix the same top 3 whole articles). The question text is the stored one, without the PII step. Faithfulness is judged against everything the model saw. Precision and recall are judged against the 6 retrieved chunks.

**Reading the numbers.**
- Faithfulness is high on both datasets, so the system rarely says things its passages do not support. Its wrong answers are mostly well supported but answer a slightly different question or only part of it, which fits what the error breakdown shows.
- Retrieval is strong. A known source document is in the first 3 articles for 94% of questions on both datasets.
- Hit rate and mean reciprocal rank need a known source document. Wix has one for 198 of 200 questions. For TechQA only 100 of the 200 questions have one, so those two rows use 100 questions.
- Answer relevancy is the one number to read with care. It compares embeddings of short questions, so it sits well below 1 for good answers. Use it to compare versions, not to judge a single value.

Context precision, context recall and answer relevancy were judged by Laguna alone. Context precision covers 179 of 200 Wix and 169 of 200 TechQA questions: on the rest the judge's output could not be parsed even after asking one chunk at a time. Everything else is complete. The scripts and per-row scores are in `eval/results/live_metrics/`.

## Speed and token cost

Both come from n8n's own execution records, which keep the timing and the model's reported token usage for every node of every request. Wix and TechQA are separate. The records cover 171 Wix and 237 TechQA successful requests that reached the answer step on the current version. That is the evaluation traffic plus a few smoke tests and repeated requests, so the counts do not match the 200 questions one to one, and the times differ a little from the client-side times in the results table.

| | Wix | TechQA |
|---|---|---|
| Whole request, median (90th percentile) | 172 s (480 s) | 195 s (724 s) |
| Answer call, median | 171 s | 193 s |
| Everything else, median | 1.7 s | 1.6 s |
| Of which: guard check on the question | 0.3 s | 0.3 s |
| Of which: embed the question | 0.7 s | 0.7 s |
| Of which: vector search | under 0.1 s | under 0.1 s |
| Of which: guard check on the reply | 0.5 s | 0.5 s |
| Input tokens per request, mean (90th percentile) | 2,903 (4,418) | 1,803 (2,083) |
| Output tokens per request, mean (90th percentile) | 728 (1,183) | 746 (1,190) |

The answer call is the only slow step, and in the evaluation it went to a shared free endpoint. Everything the pipeline itself does takes under two seconds. Wix sends more input tokens because it reads whole articles.

**Time for one row on an uncongested endpoint.** The long times above come from that shared endpoint. To time the pipeline on a normal model endpoint, 15 Wix and 15 TechQA questions went one at a time to Mistral's own API for `mistral-large-4`, with the same prompt and settings as the live workflow (reasoning effort low, temperature 0). Only the model's response was timed:

| | Wix | TechQA |
|---|---|---|
| Answer call, median | 7.9 s | 7.7 s |
| Answer call, mean (90th percentile) | 12.7 s (15.1 s) | 12.2 s (17.1 s) |
| Output speed | about 119 tokens per second | about 107 tokens per second |
| Pipeline around it (guards, embedding, search), median | 1.7 s | 1.6 s |
| One row end to end, median | about 9.6 s | about 9.3 s |
| One row end to end, mean | about 14.5 s | about 13.9 s |

The mean sits above the median because an occasional call is slow (the slowest took 74 s on Wix and 66 s on TechQA). 15 calls per dataset is a small sample, so read these as typical values. The time is almost all the model generating its reply, which averages 800 to 950 output tokens at this reasoning setting. A faster or smaller model would shorten it, and the pipeline around it is not the constraint. The script is `eval/results/live_metrics/answer_speed.py`.

**Cost.** Nothing was spent in this project: the models run on free development tiers. As an estimate for paid use, the mean token counts above give about $3.6 to $7.0 per 1,000 Wix requests and $2.8 to $5.6 per 1,000 TechQA requests. The range comes from two list prices quoted for `mistral-large-4` by third-party price trackers ($0.70 to $1.36 per million input tokens and $2.10 to $4.18 per million output tokens). The estimate leaves out the two small guard-model calls and the embedding call per request, which are short.

The script is `eval/results/live_metrics/cost_speed.py`, and its output is saved next to it as `cost_speed.md`.

## The judges

Each judge sees only the reference answer and the system's answer, and gives a 1 to 5 score for how consistent the answer is with the reference. The judges never see the documents. That means a correct answer that differs from an unfair reference is graded as wrong. This is the main weakness of the method, and the reason the unclear-label rows are listed.

None of the judges comes from the family of the answer model (Mistral), so the answer model does not grade itself.

**Agreement.** On 744 answers graded by all three, the judges give the same good-or-not-good verdict:

| Pair | Same verdict | Cohen's kappa | Same 1 to 5 score |
|---|---|---|---|
| Nemotron and GLM | 93.0% | 0.73 | 78% |
| Nemotron and Laguna | 94.0% | 0.78 | 76% |
| GLM and Laguna | 95.0% | 0.79 | 78% |

**Why Laguna.** Candidate judges from other families were tested on the same answers. Gemma-4-31B and Step-5-preview agreed reasonably but were too slow (about 9 and 6 answers per minute). Laguna graded 359 live answers in about 6 minutes and agrees with the other two judges as often as they agree with each other.

**Why three judges.** With two judges there is no tiebreaker, and the choice of rule moves the headline by up to 0.11:

| Rule on two judges (Nemotron, GLM) | Wix score | TechQA score |
|---|---|---|
| Both must say good | +0.650 | +0.665 |
| Average score at least 4 | +0.660 | +0.675 |
| Either says good | +0.790 | +0.725 |
| Three judges, 2 of 3 (used here) | +0.750 | +0.715 |

## What is not measured

- **Token counts on the live run.** The webhook does not return token usage.
- **Passages taken straight from the live run.** The retrieval metrics use passages rebuilt with the same search, not the exact text each live request saw.
- **Cost and throughput at scale.** Cost is estimated from list prices, and speed was timed on 15 clean calls per dataset. Sustained traffic was not tested.

## Reproducing the numbers

```bash
cd eval
python results/final_numbers/final_numbers.py
```

This reads the saved answers and grades in `eval/results/live_best/work/` and rewrites `eval/results/final_numbers/README.md`. The judge panel and the rule for a good answer are defined in `eval/harness/panel.py` and covered by unit tests.

The retrieval and answer-quality table comes from `eval/results/live_metrics/live_metrics.py report`, which reads the saved per-row scores in the same folder. Re-scoring needs Qdrant running and an NVIDIA API key.
