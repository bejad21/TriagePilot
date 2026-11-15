"""Build the v2 evaluation files from the frozen Wix and TechQA data, without
modifying the originals:

  wix                  eval/data/wix_v2.json
  techqa               eval/data/techqa_v2.json
  techqa-unanswerable  eval/data/techqa_unanswerable.json (downloads the parquet)

Existing output files are never overwritten.
"""
import json
import os
import sys

import pandas as pd

from prepare_techqa_corpus import TECHQA_URL, sanitize_filename

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)
DATA_DIR = os.path.join(EVAL_DIR, "data")
WIX_EVAL = os.path.join(EVAL_DIR, "wix_eval_expertwritten.json")
WIX_URL_MAP = os.path.join(EVAL_DIR, "wix_source_url_to_filename.json")
TECHQA_EVAL = os.path.join(EVAL_DIR, "techqa_eval.json")
WIX_CORPUS = os.path.join(PROJECT_ROOT, "infra", "local_files", "wix-docs")
TECHQA_CORPUS = os.path.join(PROJECT_ROOT, "infra", "local_files", "techqa-docs")

# The preparation scripts cut answers and questions at this length.
ANSWER_CUT = 1000
QUESTION_CUT = 1000


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_wix_v2(eval_json, url_map_json, corpus_dir, expected_rows=200):
    """Wix rows with gold filenames checked against the corpus folder."""
    rows = _load(eval_json)
    assert len(rows) == expected_rows, f"expected {expected_rows} Wix rows, got {len(rows)}"
    url_map = _load(url_map_json)
    out = []
    for i, row in enumerate(rows, start=1):
        urls = [u.strip() for u in row["source_url"].split(";") if u.strip()]
        gold, unmapped = [], []
        for url in urls:
            name = url_map.get(url)
            if name and os.path.isfile(os.path.join(corpus_dir, name)):
                gold.append(name)
            else:
                unmapped.append(url)
        out.append({
            "id": i,
            "question": row["question"],
            "expected_answer": row["expected_answer"],
            "answer_truncated": len(row["expected_answer"]) == ANSWER_CUT,
            "gold_sources": gold,
            "gold_urls": urls,
            "unmapped_urls": unmapped,
            "expected_behavior": "answer",
            "origin": "original",
        })
    return out


def build_techqa_v2(answerable_json, corpus_dir, expected_rows=608):
    """Answerable TechQA rows with duplicate questions marked."""
    rows = _load(answerable_json)
    assert len(rows) == expected_rows, f"expected {expected_rows} TechQA rows, got {len(rows)}"
    first_id = {}
    out = []
    for i, row in enumerate(rows, start=1):
        name = sanitize_filename(row["source_url"])
        out.append({
            "id": i,
            "question": row["question"],
            "expected_answer": row["expected_answer"],
            "answer_truncated": len(row["expected_answer"]) == ANSWER_CUT,
            "gold_sources": [name] if name and os.path.isfile(os.path.join(corpus_dir, name)) else [],
            "duplicate_of": first_id.get(row["question"]),
            "expected_behavior": "answer",
            "origin": "original",
        })
        first_id.setdefault(row["question"], i)
    return out


def fetch_techqa_parquet(url):
    """Download the TechQA-RAG-Eval parquet; the only network access here."""
    return pd.read_parquet(url)


def build_techqa_unanswerable(df, answerable_questions):
    """Published unanswerable questions, in dataset order."""
    # The frozen answerable questions were cut to QUESTION_CUT characters.
    cut = {q[:QUESTION_CUT] for q in answerable_questions}
    keep = df[(df["is_impossible"] == True)  # noqa: E712 (pandas elementwise comparison)
              & ~df["question"].str[:QUESTION_CUT].isin(cut)]
    return [{"id": i, "question": q, "gold_sources": [],
             "expected_behavior": "escalate", "origin": "published"}
            for i, q in enumerate(keep["question"], start=1)]


def write_json(path, rows):
    """Write rows as JSON, refusing to replace an existing file."""
    with open(path, "x", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    commands = {"wix": "wix_v2.json", "techqa": "techqa_v2.json",
                "techqa-unanswerable": "techqa_unanswerable.json"}
    if len(args) != 1 or args[0] not in commands:
        print("usage: build_eval_sets.py wix|techqa|techqa-unanswerable")
        return 2
    out_path = os.path.join(DATA_DIR, commands[args[0]])
    if os.path.exists(out_path):
        print(f"{out_path} already exists; refusing to overwrite")
        return 1

    if args[0] == "wix":
        rows = build_wix_v2(WIX_EVAL, WIX_URL_MAP, WIX_CORPUS)
    elif args[0] == "techqa":
        rows = build_techqa_v2(TECHQA_EVAL, TECHQA_CORPUS)
    else:
        df = fetch_techqa_parquet(TECHQA_URL)
        answerable = {r["question"] for r in _load(TECHQA_EVAL)}
        rows = build_techqa_unanswerable(df, answerable)
        impossible = int((df["is_impossible"] == True).sum())  # noqa: E712 (pandas elementwise comparison)
        print(f"total rows: {len(df)}; is_impossible: {impossible}; "
              f"after removing answerable questions: {len(rows)}")
    write_json(out_path, rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
