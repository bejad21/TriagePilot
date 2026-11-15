"""Build the two Qdrant collections the workflows search: embedded Wix articles and embedded TechQA technotes.

Run from eval/ with NVIDIA_API_KEY set and Qdrant running:

  python ingest/build_collections.py prepare            download the public datasets and write the article files (needs `pip install pandas pyarrow`)
  python ingest/build_collections.py chunks wix|techqa  split the articles and print the chunk counts (no API call)
  python ingest/build_collections.py check              compare the chunks with a reference export of the live collections
  python ingest/build_collections.py build wix|techqa   embed every chunk and load the collection (resumable)
  python ingest/build_collections.py compare wix|techqa compare search results of the new collection with a reference collection

Each chunk is embedded as "Title: <article title>\\n\\n<chunk>" with nvidia/nemotron-3.5-embed-8b (input_type passage). Only the first
2,048 of the 4,096 values are kept, re-normalised, and the collection uses cosine distance. The endpoint accepts one text per call, so
16 calls run in parallel. Vectors are cached in ingest/work/ so an interrupted build continues where it stopped.
"""
import argparse
import concurrent.futures as cf
import json
import pathlib
import random
import subprocess
import sys
import time
import uuid

import numpy as np
import requests

HERE = pathlib.Path(__file__).resolve().parent
EVAL = HERE.parent
ROOT = EVAL.parent
sys.path.insert(0, str(EVAL))
from harness.nvidia_key import api_key  # noqa: E402 (after the sys.path insert)
from ingest.chunking import chunk_article  # noqa: E402 (after the sys.path insert)

DATASETS = {
    "wix": {"dir": ROOT / "infra/local_files/wix-docs", "collection": "wix_help_center_docs", "prepare": EVAL / "prepare_wix_corpus.py"},
    "techqa": {"dir": ROOT / "infra/local_files/techqa-docs", "collection": "techqa_docs", "prepare": EVAL / "prepare_techqa_corpus.py"},
}
MODEL = "nvidia/nemotron-3.5-embed-8b"
DIMS = 2048
WORK = HERE / "work"
DEFAULT_REFERENCE = EVAL / "results/embedding_test/work"


def read_chunks(ds):
    folder = DATASETS[ds]["dir"]
    files = sorted(folder.glob("*.txt"))
    if not files:
        sys.exit(f"No article files in {folder}. Run: python ingest/build_collections.py prepare")
    chunks = []
    for f in files:
        for i, c in enumerate(chunk_article(ds, f.read_bytes().decode("utf-8"))):
            chunks.append({**c, "source": f.name, "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{ds}/{f.name}#{i}"))})
    return chunks


def embed_one(text):
    body = {"model": MODEL, "input": [text], "input_type": "passage", "truncate": "END"}
    for attempt in range(8):
        try:
            r = requests.post("https://integrate.api.nvidia.com/v1/embeddings", headers={"Authorization": "Bearer " + api_key()}, json=body, timeout=120)
            if r.status_code == 200:
                return r.json()["data"][0]["embedding"]
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return None


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def embed_all(ds, chunks, limit=None):
    WORK.mkdir(exist_ok=True)
    n = len(chunks) if limit is None else min(limit, len(chunks))
    cache = WORK / f"vectors_{ds}_{len(chunks)}.npy"
    mat = np.load(cache) if cache.exists() else np.zeros((len(chunks), DIMS), dtype=np.float32)
    todo = [i for i in range(n) if not mat[i].any()]
    print(f"{ds}: {n} chunks, {n - len(todo)} already embedded", flush=True)
    started = time.monotonic()
    with cf.ThreadPoolExecutor(16) as pool:
        futures = {pool.submit(embed_one, f"Title: {chunks[i]['title']}\n\n{chunks[i]['text']}"): i for i in todo}
        for done, fut in enumerate(cf.as_completed(futures), 1):
            v = fut.result()
            if v is not None:
                mat[futures[fut]] = unit(np.array(v[:DIMS], dtype=np.float32))
            if done % 400 == 0:
                np.save(cache, mat)
                print(f"  {done}/{len(todo)} in {time.monotonic() - started:.0f}s", flush=True)
    np.save(cache, mat)
    failed = [i for i in range(n) if not mat[i].any()]
    if failed:
        sys.exit(f"{len(failed)} chunks could not be embedded. Run the same command again to retry them.")
    return mat[:n]


def load_qdrant(url, name, chunks, mat):
    requests.delete(f"{url}/collections/{name}")
    requests.put(f"{url}/collections/{name}", json={"vectors": {"size": DIMS, "distance": "Cosine"}}).raise_for_status()
    for i in range(0, len(mat), 200):
        points = [{"id": c["id"], "vector": mat[i + j].tolist(), "payload": {k: c[k] for k in ("text", "source", "title", "line")}} for j, c in enumerate(chunks[i:i + 200])]
        requests.put(f"{url}/collections/{name}/points?wait=true", json={"points": points}).raise_for_status()
    count = requests.get(f"{url}/collections/{name}").json()["result"]["points_count"]
    print(f"{name}: {count} points loaded")


def cmd_prepare(_):
    for ds, cfg in DATASETS.items():
        print("preparing", ds, flush=True)
        subprocess.run([sys.executable, str(cfg["prepare"])], cwd=str(EVAL), check=True)


def cmd_chunks(args):
    chunks = read_chunks(args.dataset)
    sources = {c["source"] for c in chunks}
    print(f"{args.dataset}: {len(sources)} articles, {len(chunks)} chunks")


def cmd_check(args):
    ok = True
    for ds in DATASETS:
        ref_file = pathlib.Path(args.reference) / f"chunks_{ds}.json"
        if not ref_file.exists():
            print(f"{ds}: reference file {ref_file} not found, skipped")
            continue
        ref = json.loads(ref_file.read_text(encoding="utf-8"))
        mine = read_chunks(ds)
        a = sorted((c["source"], c["text"]) for c in ref)
        b = sorted((c["source"], c["text"]) for c in mine)
        same = a == b
        titles = {c["source"]: c["title"] for c in ref} == {c["source"]: c["title"] for c in mine}
        ok &= same and titles
        print(f"{ds}: reference {len(a)} chunks, rebuilt {len(b)} chunks, identical text and source: {same}, identical titles: {titles}")
    sys.exit(0 if ok else 1)


def cmd_build(args):
    chunks = read_chunks(args.dataset)
    if args.limit:
        chunks = chunks[:args.limit]
    mat = embed_all(args.dataset, chunks, None)
    load_qdrant(args.qdrant, DATASETS[args.dataset]["collection"] + args.suffix, chunks, mat)


def search(url, name, vector, k=6):
    r = requests.post(f"{url}/collections/{name}/points/search", json={"vector": vector, "limit": k, "with_payload": True}).json()["result"]
    return [((p["payload"]["source"], p["payload"]["text"]), p["score"]) for p in r]


def cmd_compare(args):
    new = DATASETS[args.dataset]["collection"] + args.suffix
    ref_url = args.reference_qdrant or args.qdrant
    ref = args.reference_collection or DATASETS[args.dataset]["collection"] + "_nemo"
    pts, offset = [], None
    while True:
        body = {"limit": 1000, "with_vector": True, "with_payload": False, **({"offset": offset} if offset else {})}
        res = requests.post(f"{args.qdrant}/collections/{new}/points/scroll", json=body).json()["result"]
        pts += res["points"]
        offset = res.get("next_page_offset")
        if not offset:
            break
    sample = random.Random(0).sample(pts, min(args.n, len(pts)))
    same_set = same_order = top1 = 0
    sims = []
    for p in sample:
        mine, theirs = search(args.qdrant, new, p["vector"]), search(ref_url, ref, p["vector"])
        a, b = [x[0] for x in mine], [x[0] for x in theirs]
        same_set += set(a) == set(b)
        same_order += a == b
        top1 += a[0] == b[0]
        sims.append(theirs[0][1])
    print(f"{args.dataset}: {new} against {ref}, {len(sample)} queries = vectors of {len(sample)} random chunks")
    print(f"  the closest reference chunk is the same chunk: {top1} of {len(sample)}; mean cosine to it {np.mean(sims):.4f} (1.0000 = identical vectors)")
    print(f"  top 6 hold the same chunks: {same_set} of {len(sample)}; same chunks in the same order: {same_order} of {len(sample)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").set_defaults(fn=cmd_prepare)
    for name, fn in (("chunks", cmd_chunks), ("build", cmd_build), ("compare", cmd_compare)):
        s = sub.add_parser(name)
        s.add_argument("dataset", choices=list(DATASETS))
        s.add_argument("--qdrant", default="http://localhost:6333")
        s.add_argument("--suffix", default="_nemo", help="appended to the collection name (default _nemo, the name the workflows search)")
        s.add_argument("--limit", type=int, help="build only the first N chunks (for a quick test)")
        s.add_argument("--n", type=int, default=20, help="compare: number of sample queries")
        s.add_argument("--reference-collection", help="compare: the collection to compare with (default: the live <name>_nemo)")
        s.add_argument("--reference-qdrant", help="compare: Qdrant URL holding the reference collection (default: same as --qdrant)")
        s.set_defaults(fn=fn)
    c = sub.add_parser("check")
    c.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="folder holding chunks_wix.json and chunks_techqa.json exported from the live collections")
    c.set_defaults(fn=cmd_check)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
