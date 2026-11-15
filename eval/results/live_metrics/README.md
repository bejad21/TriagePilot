# RAG metrics of the live run (Wix and TechQA separately)

Built by `live_metrics.py`; method in its docstring. 95% intervals from resampling the rows.

| Metric | Wix | TechQA |
|---|---|---|
| Faithfulness (mean of two judges) | 0.98 (0.98 to 0.99) (n=190) | 0.95 (0.93 to 0.96) (n=169) |
|   Nemotron judge | 0.98 (0.97 to 0.99) (n=189) | 0.94 (0.92 to 0.96) (n=169) |
|   Laguna judge | 0.99 (0.98 to 1.00) (n=190) | 0.95 (0.94 to 0.97) (n=169) |
| Answer relevancy (Laguna judge) | 0.62 (0.60 to 0.63) (n=190) | 0.58 (0.56 to 0.60) (n=169) |
| Context precision, 6 retrieved chunks (Laguna judge) | 0.89 (0.85 to 0.92) (n=179) | 0.88 (0.84 to 0.91) (n=169) |
| Context recall, 6 retrieved chunks (Laguna judge) | 0.84 (0.80 to 0.87) (n=200) | 0.87 (0.83 to 0.91) (n=200) |
| Hit rate @ 3 articles | 0.94 (0.91 to 0.97) (n=198) | 0.94 (0.89 to 0.98) (n=100) |
| Hit rate @ 6 chunks | 0.94 (0.91 to 0.97) (n=198) | 0.95 (0.90 to 0.99) (n=100) |
| Mean reciprocal rank | 0.82 (0.78 to 0.87) (n=198) | 0.92 (0.86 to 0.97) (n=100) |
