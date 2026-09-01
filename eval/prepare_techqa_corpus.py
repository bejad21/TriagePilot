"""Downloads the real TechQA-RAG-Eval benchmark (nvidia/TechQA-RAG-Eval on Hugging Face,
Apache 2.0, https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval - a reduced, RAG-ready
cut of IBM Research's original TechQA dataset) and prepares two things TriagePilot needs
for a third, real domain proving the system generalizes beyond Chatwoot and Wix:

1. The real IBM Technotes cited as ground truth by the dataset's answerable questions,
   written to infra/local_files/techqa-docs/ for the "Ingest TechQA Docs" n8n workflow to
   chunk and embed into Qdrant. Every question in this dataset is answerable from exactly
   one real Technote, so ingesting only the cited set (496 real documents, no sampling or
   padding needed) already gives every eval question a genuine, retrievable source - unlike
   Wix's corpus, which needed padding to reach a comparable scale.
2. The real question/answer pairs (genuine questions asked on IBM's developer support
   forums, with answers grounded in a published Technote - not synthetic Q&A) as eval rows,
   matching the same {question, expected_answer, source_url} shape used throughout this
   project's other eval datasets. `source_url` here is the cited Technote's filename, not a
   public URL - this reduced dataset doesn't carry the original IBM support-site URLs.

Only the answerable rows (is_impossible == False) are used, matching WixQA-ExpertWritten's
pattern of real, ground-truth-bearing questions. The dataset also includes 300 deliberately
unanswerable questions (no Technote actually answers them) for training readers not to
hallucinate - out of scope for this eval, which measures answer quality, not refusal
behavior.

The corpus is not committed to git (infra/local_files/ is gitignored, same policy as the
Chatwoot and Wix corpora) since it's fully reproducible by running this script - no scraping,
no auth, no rate limits.
"""
import json
import os
import re

import pandas as pd

TECHQA_URL = ("https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval/"
              "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet")


def sanitize_filename(name):
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-")


def extract_unique_documents(answerable):
    docs = {}
    for ctxs in answerable["contexts"]:
        for c in ctxs:
            docs.setdefault(c["filename"], c["text"])
    return docs


def write_corpus_files(docs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for filename, text in docs.items():
        path = os.path.join(out_dir, sanitize_filename(filename))
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def build_eval_rows(answerable):
    rows = []
    for _, row in answerable.iterrows():
        if not row["answer"]:
            continue
        rows.append({
            "question": row["question"][:1000],
            "expected_answer": row["answer"][:1000],
            "source_url": row["contexts"][0]["filename"] if row["contexts"] else "",
        })
    return rows


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus_dir = os.path.join(project_root, "infra", "local_files", "techqa-docs")
    eval_out_path = os.path.join(os.path.dirname(__file__), "techqa_eval.json")

    df = pd.read_parquet(TECHQA_URL)
    answerable = df[df["is_impossible"] == False]  # noqa: E712

    docs = extract_unique_documents(answerable)
    write_corpus_files(docs, corpus_dir)
    print(f"Wrote {len(docs)} real IBM Technote corpus files to {corpus_dir}")

    eval_rows = build_eval_rows(answerable)
    json.dump(eval_rows, open(eval_out_path, "w", encoding="utf-8"))
    print(f"Wrote {len(eval_rows)} real eval rows to {eval_out_path}")


if __name__ == "__main__":
    main()
