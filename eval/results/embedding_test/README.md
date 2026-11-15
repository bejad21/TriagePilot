# Embedding test: shared files

This folder holds only files that other scripts use: `queries.py` (question forms and query vectors), and under `work/` the saved passages, the saved answers and grades of one arm, and the chunk export (`chunks_wix.json`, `chunks_techqa.json`) that `python ingest/build_collections.py check` compares against.

The experiment itself is summarised in [docs/experiments/embedding-model.md](../../../docs/experiments/embedding-model.md).
