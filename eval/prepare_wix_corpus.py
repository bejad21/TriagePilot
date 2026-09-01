"""Downloads the real WixQA benchmark (Wix/WixQA on Hugging Face, MIT licensed,
https://huggingface.co/datasets/Wix/WixQA) and prepares two things TriagePilot
needs for a second, real domain proving the system generalizes beyond Chatwoot:

1. A bounded corpus of real Wix help center articles, written to
   infra/local_files/wix-docs/ for the "Ingest Wix Docs" n8n workflow
   (workflows/06-ingest-wix-docs.json) to chunk and embed into Qdrant.
2. The real 200 ExpertWritten question/answer pairs (genuine customer support
   tickets with expert-reviewed answers - the most authentic subset of the
   three WixQA configs) as eval rows, matching the same
   {question, expected_answer, source_url} shape used throughout this project's
   other eval datasets.

The corpus is not committed to git (infra/local_files/ is gitignored, same
policy as the Chatwoot docs checkout) since it's fully reproducible by running
this script - it re-downloads WixQA's public parquet files, no scraping,
no auth, no rate limits.

Scope note: the full wix_kb_corpus config has 4,109 real "article"-type
entries (6,221 total across article/feature_request/known_issue types). This
script selects every article cited as ground truth by an ExpertWritten
question, plus a random sample bringing the total to ~1,200 - a scale
deliberately comparable to Chatwoot's ~590-point corpus rather than the full
4,109, to keep ingestion time bounded. Increase SAMPLE_SIZE below to ingest
more.
"""
import json
import os
import random
import re

import pandas as pd

SAMPLE_SIZE = 1200
SEED = 42

HF_BASE = "https://huggingface.co/datasets/Wix/WixQA/resolve/refs%2Fconvert%2Fparquet"
EXPERTWRITTEN_URL = f"{HF_BASE}/wixqa_expertwritten/train/0000.parquet"
KB_CORPUS_URL = f"{HF_BASE}/wix_kb_corpus/train/0000.parquet"


def slugify(title, article_id):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title or "untitled").strip("-").lower()[:80]
    return f"{slug}-{article_id[:8]}.txt"


def select_corpus_articles(kb, cited_ids, sample_size, seed):
    articles = kb[kb["article_type"] == "article"]
    cited_articles = articles[articles["id"].isin(cited_ids)]
    rest = articles[~articles["id"].isin(cited_ids)]
    pad_n = max(0, sample_size - len(cited_articles))
    padding = rest.sample(n=min(pad_n, len(rest)), random_state=seed)
    return pd.concat([cited_articles, padding])


def write_corpus_files(selected, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for _, row in selected.iterrows():
        path = os.path.join(out_dir, slugify(row["title"], row["id"]))
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {row['title']}\n\n{row['contents']}")


def build_url_to_filename_map(selected):
    return {row["url"]: slugify(row["title"], row["id"]) for _, row in selected.iterrows()}


def build_eval_rows(qa, kb):
    kb_url = dict(zip(kb["id"], kb["url"]))
    rows = []
    for _, row in qa.iterrows():
        urls = [kb_url.get(aid, "") for aid in row["article_ids"]]
        rows.append({
            "question": row["question"][:1000],
            "expected_answer": row["answer"][:1000],
            "source_url": "; ".join(u for u in urls if u)[:500],
        })
    return rows


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus_dir = os.path.join(project_root, "infra", "local_files", "wix-docs")
    eval_out_path = os.path.join(os.path.dirname(__file__), "wix_eval_expertwritten.json")

    qa = pd.read_parquet(EXPERTWRITTEN_URL)
    kb = pd.read_parquet(KB_CORPUS_URL)

    cited_ids = set()
    for ids in qa["article_ids"]:
        cited_ids.update(ids)

    selected = select_corpus_articles(kb, cited_ids, SAMPLE_SIZE, SEED)
    write_corpus_files(selected, corpus_dir)
    print(f"Wrote {len(selected)} corpus files to {corpus_dir} "
          f"({len(cited_ids)} unique articles cited by eval questions, "
          f"{sum(selected['id'].isin(cited_ids))} of those covered)")

    url_map = build_url_to_filename_map(selected)
    url_map_path = os.path.join(os.path.dirname(__file__), "wix_source_url_to_filename.json")
    json.dump(url_map, open(url_map_path, "w", encoding="utf-8"))
    print(f"Wrote {len(url_map)} source_url -> filename mappings to {url_map_path}")

    eval_rows = build_eval_rows(qa, kb)
    json.dump(eval_rows, open(eval_out_path, "w", encoding="utf-8"))
    print(f"Wrote {len(eval_rows)} real eval rows to {eval_out_path}")


if __name__ == "__main__":
    main()
