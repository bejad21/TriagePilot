# Hybrid search and reranker tests

Experiments 6 and 7 in [docs/experiments.md](../../../docs/experiments.md), and the choice of Laguna as a judge.

`rr_hybrid.py` runs the experiment on 100 rows (50 Wix and 50 TechQA, with many wrong and handed-off cases). For each search version it builds the retrieved passages, answers the rows with the same model and prompt as the live system, and grades the answers with three judges. `python results/rerank_hybrid/rr_hybrid.py report` prints the comparison from the saved files.

| Search version | What it is |
|---|---|
| H | Hybrid: keyword search (BM25) and dense search combined by reciprocal rank fusion, top 6 chunks |
| RR | Dense top 30, reranked with Cohere rerank-v3.5, top 6 |
| HR | Hybrid top 30, reranked with Cohere, top 6 |
| B2 | The dense baseline run a second time on the same 100 rows (noise control) |
| HT | Hybrid plus Cohere rerank on the other 50 of the original 100 TechQA rows |
| HX | Hybrid alone on the 150 Wix rows outside the first 100-row test |

| File | Contents |
|---|---|
| `rr_hybrid.py` | The experiment. Its docstring lists the subcommands. |
| `laguna_full.py` | Grades every saved answer with Laguna and prints how it agrees with the other judges (`report`). |
| `judge_candidates.py`, `nv_judge_candidates.py` | Tests of other models as judges on 60 answers. |
| `work/gen_*.jsonl` | The saved answers per search version. |
| `work/judge_*.jsonl` | The judge grades per search version. |
| `work/arm_contexts.json` | The retrieved passages of every search version (7 MB). |
| `work/nv_*`, `work/cand_*`, `work/laguna_full.jsonl` | The outputs of the judge tests. |
