# Embedding model test

Experiment 2 in [experiments](../experiments.md). The question: does a stronger embedding model, with the article title added to each chunk, find the right article more often? It does. This page holds the final table from the saved run outputs. The scripts and raw files of this run are not part of this repository.

300 questions (200 Wix, 100 TechQA), exact cosine search over the stored chunks. `art@3` is the number of questions whose source article is among the top 3 distinct articles. `chunk@15` is the same for the top 15 chunks, and MRR is the mean reciprocal rank of the source article.

| Setting | Wix art@3 | Wix chunk@15 | Wix MRR | TechQA art@3 | TechQA chunk@15 |
|---|---|---|---|---|---|
| Previous embedding model, customer question | 165 | 193 | 0.70 | 89 | 96 |
| Nemotron 8B plain, 4,096 dimensions, question | 176 | 195 | 0.76 | 94 | 97 |
| **Nemotron 8B + title, 2,048 dimensions, question** | **187** | 196 | **0.81** | 94 | 97 |
| Nemotron 8B + title, 1,024 dimensions, question | 186 | 193 | 0.80 | 94 | 96 |
| Nemotron 8B + title, model-written rewrite only | 175 | 190 | 0.76 | 92 | 95 |
| Nemotron 8B + title, question plus hypothetical answer (sum) | 187 | 195 | 0.81 | 96 | 97 |

- The best simple setting is the article title in front of each chunk, 2,048 dimensions and the customer's own question.
- Rewriting the question with a model hurts on Wix. A hypothetical answer adds at most one question, for one extra model call per question.
- The source article was already in the top 15 chunks for 96 to 97% of questions, so the gain is in ranking.
- Final answers moved only within noise (0.71 to 0.74 on a small answer model, 0.75 to 0.75 on `mistral-large-4`, over 300 questions), and tokens per request fell 18%.

Settings: `input_type` is `query` for questions and `passage` for chunks, `truncate: END`, the title added as `Title: <article title>` before the chunk text, chunking unchanged at 1,000 characters with 200 overlap.
