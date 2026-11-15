"""Held-out row selection and cross-domain abstention sets. Question text is never edited."""

import json
import random
from collections.abc import Collection, Sequence
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TUNING_IDS = range(1, 31)


def heldout_rows(rows: Sequence[dict], start: int = 31, stop: int = 130) -> list[dict]:
    """Answerable, non-duplicate rows with start <= id <= stop, ordered by id."""
    kept = [
        r
        for r in rows
        if start <= r["id"] <= stop
        and r["expected_behavior"] == "answer"
        and not r.get("duplicate_of")
    ]
    return sorted(kept, key=lambda r: r["id"])


def assert_not_tuned(rows: Sequence[dict], tuned_max: int = 30) -> None:
    """Raise ValueError naming every row id <= tuned_max (rows the system was tuned on)."""
    tuned = sorted(r["id"] for r in rows if r["id"] <= tuned_max)
    if tuned:
        raise ValueError(f"tuned row ids must not enter a held-out run: {tuned}")


def cross_domain_rows(
    source_rows: Sequence[dict],
    source_dataset: str,
    n: int,
    seed: int = 0,
    exclude_ids: Collection[int] = (),
) -> list[dict]:
    """Sample n answerable rows and relabel them as rows the other system must escalate."""
    excluded = set(exclude_ids)
    pool = sorted(
        (r for r in source_rows if r["id"] not in excluded and r["expected_behavior"] == "answer"),
        key=lambda r: r["id"],
    )
    if len(pool) < n:
        raise ValueError(f"need {n} rows but only {len(pool)} available")
    picked = random.Random(seed).sample(pool, n)
    return [
        {
            "id": i,
            "question": r["question"],
            "gold_sources": [],
            "expected_behavior": "escalate",
            "origin": "mechanical",
            "source_dataset": source_dataset,
            "source_id": r["id"],
        }
        for i, r in enumerate(picked, start=1)
    ]


def write_new_json(path: Path, rows: list[dict], created: list[Path]) -> None:
    """Write rows as JSON, failing if the file exists; add the path to created once this call has opened it."""
    with open(path, "x", encoding="utf-8") as f:
        created.append(path)
        f.write(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


def write_cross_sets(data_dir: Path = DATA_DIR, n: int = 100, seed: int = 0) -> None:
    """Write both cross-domain files, refusing to overwrite and never leaving a partial result.

    Questions come from the full-length reference files, not the v2 files whose
    long questions are cut at 1000 characters; ids and selection are the same.
    """
    jobs = [
        ("wix_full.json", "wix", "wix_to_techqa_cross.json", False),
        ("techqa_full.json", "techqa", "techqa_to_wix_cross.json", True),
    ]
    for _, _, out_name, _ in jobs:
        if (data_dir / out_name).exists():
            raise FileExistsError(f"{data_dir / out_name} already exists; refusing to overwrite")
    outputs = []
    for in_name, dataset, out_name, drop_duplicates in jobs:
        rows = json.loads((data_dir / in_name).read_text(encoding="utf-8"))
        if drop_duplicates:
            rows = [r for r in rows if not r.get("duplicate_of")]
        outputs.append(
            (data_dir / out_name, cross_domain_rows(rows, dataset, n, seed, exclude_ids=TUNING_IDS))
        )
    created = []
    try:
        for path, cross in outputs:
            write_new_json(path, cross, created)
    except BaseException:
        for path in created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


if __name__ == "__main__":
    write_cross_sets()
