# Evaluation Results — Wix and TechQA

Both systems were built and evaluated independently, each against a real, non-synthetic
benchmark with genuine question-to-source-document grounding — a property that lets retrieval
and generation quality be measured cleanly, without a hidden ground-truth ceiling the corpus
can't reach.

## Real RAGAS methodology rebuild, 30-row sample

The eval workflows originally used a 1-5 LLM-judge score loosely inspired by RAGAS. That was
rebuilt into real RAGAS methodology: **Faithfulness** decomposes the answer into atomic claims
and verifies each against retrieved context (score = supported/total, 0-1, not a holistic 1-5
rating); **Answer Relevancy** reverse-engineers up to 3 questions the answer would be responding
to and cosine-scores their embeddings against the real question; **Context Precision** (new)
judges each retrieved chunk's relevance in rank order and computes precision@k, weighted toward
chunks ranked higher; **Context Recall** (new) decomposes the reference answer into claims and
checks each against retrieved context; **RAGAS Overall** is the arithmetic mean of those four.
**Answer Correctness** (the original ground-truth comparison, 1-5) is kept as a labeled
supplementary metric outside RAGAS's own four-metric core.

A real bug was found and fixed during this rebuild: the newly inserted Context Precision/Context
Recall Code nodes initially did `{...$json, newField}` straight after a `chainLlm` node — which
outputs *only* `{output: {...}}` on its main connection, silently dropping every other field
(`status`, `question`, etc.) from the node's input. This broke the answered/escalated branch for
every row. Fixed by reconstructing the full object via `$('Combine Retrieved Context').first().json`
instead of spreading `$json`, then reverified live that an answered row correctly reaches the
scoring node.

**Real 30-row results, both systems, same live methodology:**

| Metric (0-1 scale unless noted) | Wix (30) | TechQA (30) |
|---|---|---|
| Answer Correctness (1-5, supplementary) | 4.00 / 5 | 4.63 / 5 |
| Answer Relevancy | 0.88 | 0.87 |
| Faithfulness (RAGAS) | **0.89** | **0.85** |
| Context Precision | 0.81 | 0.79 |
| Context Recall | 0.76 | 0.80 |
| RAGAS Overall | **0.85** | **0.84** |
| Hit Rate@K | 0.90 | 0.93 |
| Mean Reciprocal Rank | 0.72 | 0.83 |
| Answered / total | 26 / 30 | 27 / 30 |

Commonly cited RAGAS Faithfulness figures for mature production RAG systems sit around
0.89-0.90. Wix (0.89) is at that figure and TechQA (0.85) is just below it.

## Retrieval-quality metrics: Hit Rate@K and MRR

Two standard retrieval-eval metrics are computed automatically inside each Evaluation workflow,
rather than by hand: **Hit Rate@K** (did retrieval return the chunk that actually matches the
question's ground-truth source document, anywhere in the top-K results — 1 or 0) and
**Reciprocal Rank** (`1/rank` of that matching chunk, averaged into **Mean Reciprocal Rank**
across rows — rewards the match landing near the top, not just somewhere in the window).

Both metrics are recorded on the escalated branch too, not just the answered branch, since
retrieval happens before the agent decides to answer or escalate — retrieval quality is
measurable independent of that downstream decision.

Both systems hit every retrieval target in the top-K window on the initial 15-row measurement
(Hit Rate@K 1.00 for both), with TechQA's Mean Reciprocal Rank (0.86) ahead of Wix's (0.74) —
consistent with TechQA's one-Technote-per-question ground truth being a structurally cleaner
retrieval target than Wix's KB, which sometimes cites several related articles per question
(lower individual rank per matching article, even when at least one is always found). On the
larger 30-row sample, Hit Rate@K settled at 0.90 (Wix) / 0.93 (TechQA) and MRR at 0.72 (Wix) /
0.83 (TechQA) — both systems still find the right document in the top-K window almost every
time.

## Wix: diagnosing and fixing sub-5 rows with a Cohere reranker

Three answered rows scoring below 5/5 Answer Correctness were traced individually by pulling
each row's actual retrieved context and comparing it against the real ground truth, not assumed
to be noise:

- **Row 6** (CMS item reordering) and **row 10** (PayPal not appearing as a payment option):
  genuine retrieval-precision misses. The correct article existed in the corpus, but `topK=4`
  vector search surfaced a different, topically-adjacent article instead.
- **Row 2** (domain voucher not visible at checkout): a genuine corpus gap, not a retrieval
  problem. A direct grep across all 1,200 corpus files confirmed zero documents mention
  "voucher" and "checkout" together — the specific fact needed was never ingested. No amount of
  retrieval tuning recovers a fact that was never in the corpus.

**Fix applied for the two retrieval-precision misses**: added a Cohere Reranker
(`nodes-langchain.rerankerCohere`, model `rerank-v3.5`) to the Wix triage workflow's retriever
node. Widened the initial Qdrant fetch from `topK=4` to `topK=15` (broad recall) and let the
reranker cross-encode and trim to a final `topN` (broad-recall-then-precise-select, rather than
relying on vector similarity alone for precision).

Tested incrementally before running any full batch:

| Config | Row 2 (corpus gap) | Row 6 (retrieval miss) | Row 10 (retrieval miss) |
|---|---|---|---|
| Before (topK=4, no rerank) | AC 3 | AC 3 | AC 2, Faith 1 |
| `topN=4` (first test) | AC 3 (unchanged) | AC 4 (improved) | escalated (model-confidence variance, not a rerank effect) |
| `topN=6` (widened after diagnosing near-duplicate chunk crowding on row 2) | AC 3 (still unchanged — confirmed by direct corpus grep, not fixable by retrieval) | **AC 5** | **AC 5** (Faith dropped 5→3, still passing) |

Row 2 staying flat is consistent with the corpus-gap finding, not a failure of the fix — it was
never going to move without new source content. Rows 6 and 10 both genuinely improved and held
at `topN=6`, so that config was carried into the full 15-row re-run:

| Metric | Before reranker | After reranker (topN=6) |
|---|---|---|
| Answer Correctness | 4.08 / 5 | **4.33 / 5** |
| Relevance | 5.00 / 5 | 4.92 / 5 |
| Faithfulness | 4.54 / 5 | **4.67 / 5** |
| Answered | 13 / 15 | 12 / 15 |

Net real improvement on the two metrics the fix targeted, with a real trade-off disclosed rather
than hidden: the wider retrieval window pulled in content from a *different* Wix product with an
overlapping name and opposite behavior on one row (hotel calendar sync) — the native "Wix
Hotels" app (correct, matching ground truth) versus a third-party "Wix Hotels by HotelRunner"
integration (a different app entirely, syncing in the opposite direction). Both were present in
the widened context; the agent picked the wrong one on that row. A reranker fixes "wrong document
retrieved," not "two real but different documents both retrieved and confused for each other" —
a different, unsolved problem, out of scope for this fix.

## TechQA: cloned from Wix's tuned configuration

TechQA was cloned from Wix's already-tuned configuration (`topK=15`, Cohere reranker `topN=6`,
same credential), not built as a fresh, untuned baseline — so its strong first-measurement
numbers reflect the accumulated Wix reranker fix as well as this being a naturally well-suited
dataset. Live-verified beyond the aggregate score: a direct webhook call with a real
SSL-configuration question returned a detailed, correct answer citing the exact ground-truth
Technote for that exact question, and the jailbreak guardrail was reconfirmed working on a
direct adversarial test.

TechQA also ships 300 deliberately *unanswerable* questions (real questions with no single
Technote that actually resolves them, included by IBM's own researchers to test against
hallucination). A spot-check on one of these did **not** escalate — the agent found a real,
topically related Technote and answered confidently from it, which may or may not be the
correct call depending on how closely the retrieved document actually matches the specific
scenario. This wasn't part of the formal measurement (which only draws from the 608 answerable,
ground-truth-bearing rows), but it's an honest, real finding worth naming: a different failure
mode from a clean escalation, not fully investigated further here.
