"""Build full-length reference files for the v2 Wix and TechQA sets, without
modifying the originals:

  wix     eval/data/wix_full.json     (answers restored from WixQA expertwritten)
  techqa  eval/data/techqa_full.json  (answers and questions restored from TechQA-RAG-Eval)

The v2 files inherited a 1000-character cut from the old preparation scripts.
Existing output files are never overwritten.
"""
import os
import sys

import pandas as pd

from build_eval_sets import ANSWER_CUT, DATA_DIR, QUESTION_CUT, _load, write_json
from prepare_techqa_corpus import TECHQA_URL
from prepare_wix_corpus import EXPERTWRITTEN_URL


def fetch_parquet(url):
    """Download a parquet file; the only network access here."""
    return pd.read_parquet(url)


def merge_full_text(rows, upstream, *, question_col="question", answer_col="answer"):
    """Copies of rows with the full upstream question and answer restored."""
    by_question = {}
    for question, answer in zip(upstream[question_col], upstream[answer_col]):
        by_question.setdefault(question[:QUESTION_CUT], []).append((question, answer))

    out = []
    for row in rows:
        candidates = by_question.get(row["question"][:QUESTION_CUT], [])
        if len(candidates) > 1:
            candidates = [c for c in candidates
                          if c[1][:ANSWER_CUT] == row["expected_answer"][:ANSWER_CUT]]
        # Upstream can repeat a row verbatim; identical text offers no choice to guess.
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) != 1:
            raise ValueError(f"row {row['id']}: {len(candidates)} upstream matches, expected 1")
        question, answer = candidates[0]
        merged = dict(row)
        merged["question"] = question
        merged["expected_answer"] = answer
        merged["answer_was_truncated_in_v2"] = answer != row["expected_answer"]
        merged["answer_truncated"] = False
        out.append(merged)
    return out


def _write_new(path, rows):
    """Write a new file; remove it again if the write fails part way."""
    try:
        write_json(path, rows)
    except FileExistsError:
        raise
    except BaseException:
        os.remove(path)
        raise


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    commands = {"wix": ("wix_v2.json", "wix_full.json"),
                "techqa": ("techqa_v2.json", "techqa_full.json")}
    if len(args) != 1 or args[0] not in commands:
        print("usage: build_full_references.py wix|techqa")
        return 2
    src_name, out_name = commands[args[0]]
    out_path = os.path.join(DATA_DIR, out_name)
    if os.path.exists(out_path):
        print(f"{out_path} already exists; refusing to overwrite")
        return 1

    rows = _load(os.path.join(DATA_DIR, src_name))
    if args[0] == "wix":
        upstream = fetch_parquet(EXPERTWRITTEN_URL)
    else:
        upstream = fetch_parquet(TECHQA_URL)
        upstream = upstream[upstream["is_impossible"] == False]  # noqa: E712 (pandas elementwise comparison)
    merged = merge_full_text(rows, upstream)

    answers = sum(m["answer_was_truncated_in_v2"] for m in merged)
    questions = sum(m["question"] != r["question"] for m, r in zip(merged, rows))
    longest = max(len(m["expected_answer"]) for m in merged)
    _write_new(out_path, merged)
    print(f"Wrote {len(merged)} rows to {out_path}; answers changed: {answers}; "
          f"questions changed: {questions}; longest answer: {longest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
