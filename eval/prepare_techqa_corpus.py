"""Downloads TechQA-RAG-Eval (nvidia/TechQA-RAG-Eval on Hugging Face, Apache 2.0,
https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval), a reduced, RAG-ready cut of
IBM Research's TechQA, and prepares the TechQA domain:

1. The IBM Technotes cited as ground truth by the answerable questions, written to
   infra/local_files/techqa-docs/ for the "Ingest TechQA Docs" n8n workflow to chunk
   and embed into Qdrant. Each question maps to exactly one Technote, so ingesting only
   the cited set (496 documents) needs no sampling or padding, unlike Wix.
2. The question/answer pairs (questions from IBM's developer support forums, answers
   grounded in a published Technote) as eval rows, in the same
   {question, expected_answer, source_url} shape as the other eval datasets.
   `source_url` is the Technote filename, since this cut doesn't carry the original
   IBM support-site URLs.

Only answerable rows (is_impossible == False) are used. The 300 unanswerable questions
are out of scope: this eval measures answer quality, not refusal behavior.

The corpus is gitignored (infra/local_files/) because this script reproduces it with no
scraping, auth or rate limits.
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
