"""RAG metrics for the live run, Wix and TechQA reported separately.
The webhook returns the answer but not its passages, so they are rebuilt with the live workflow's search: the stored question (without the live PII step) is embedded with nemotron-3.5-embed-8b
(query, first 2,048 dimensions, re-normalized), the live Qdrant collection returns the 6 closest chunks, and Wix reads the top 3 articles whole (12,000 characters each at most, 60,000 in total).

Metrics (RAGAS-style, definitions in harness/judges.py):
  faithfulness       share of the answer's claims that the passages the model saw support (judges: Nemotron and Laguna)            answered rows
  answer relevancy   mean cosine similarity between the question and up to 3 questions the judge writes back from the answer     answered rows
  context precision  how early the relevant chunks come among the 6 retrieved chunks (judged against the reference)              all rows (search runs before the decision to answer)
  context recall     share of the reference answer's claims that the 6 retrieved chunks contain                                   all rows
  hit@3, hit@6, MRR  whether and how early a gold source document appears in the search (rows with a known source document)      rows with gold
usage (from eval/, NVIDIA key loaded, Qdrant running): python results/live_metrics/live_metrics.py contexts | run NAME | report
NAME = faith_nemotron | faith_laguna | relevancy | precision | recall"""
import concurrent.futures as cf
import importlib.util
import json
import pathlib
import random
import statistics
import sys
import threading
import time

import numpy as np
import requests

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent
sys.path.insert(0, str(HERE.parents[1]))
from harness import judges, nvidia_key  # noqa: E402
from harness.backends import OpenAICompatChat  # noqa: E402

cmd = sys.argv[1]
_argv = sys.argv[:]
sys.argv = ["x", "none"]
spec = importlib.util.spec_from_file_location("live_eval", RES / "live_best/live_eval.py")
le = importlib.util.module_from_spec(spec)
spec.loader.exec_module(le)
spec2 = importlib.util.spec_from_file_location("queries", RES / "embedding_test/queries.py")
qmod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(qmod)
sys.argv = _argv
LIVEW = RES / "live_best/work"
FILES = HERE.parents[2] / "infra/local_files"
NV = "https://integrate.api.nvidia.com/v1"
COLL = {"wix": "wix_help_center_docs_nemo", "techqa": "techqa_docs_nemo"}
CTX = HERE / "contexts.json"


def jl(p):
    return le.jl(p)


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-9)


def window(text, hits, cap=12000):
    if len(text) <= cap:
        return text
    pos = [p for p in (text.find(h[:120]) for h in hits) if p >= 0]
    lo, hi = (min(pos), max(pos) + 1000) if pos else (0, 0)
    lo = max(0, lo - max(0, (cap - (hi - lo)) // 2))
    return text[lo:lo + cap]


def build_context(ds, top):
    if ds == "wix":
        arts = list(dict.fromkeys(c["source"] for c in top))[:3]
        parts = []
        for s in arts:
            txt = (FILES / "wix-docs" / s).read_text(encoding="utf-8", errors="ignore")
            parts.append(f"[{s}]\n" + window(txt, [c["text"] for c in top if c["source"] == s]))
        parts += [f"[{c['source']}]\n{c['text']}" for c in top if c["source"] not in arts]
        return "\n\n---\n\n".join(parts)[:60000]
    return "\n\n---\n\n".join(f"[{c['source']}]\n{c['text']}" for c in top)


def embed(texts):
    for attempt in range(6):
        r = requests.post(f"{NV}/embeddings", headers={"Authorization": "Bearer " + nvidia_key.api_key()}, timeout=120,
                          json={"model": "nvidia/nemotron-3.5-embed-8b", "input": [t[:6000] for t in texts], "input_type": "query", "truncate": "END"})
        if r.status_code == 200:
            return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
        time.sleep(10 if r.status_code == 429 else 4)
    raise RuntimeError("embedding failed")


class NvEmbedder:
    def embed(self, texts):
        return [embed([t])[0] for t in texts]  # this endpoint takes one text per call


def contexts():
    R = le.rows()
    gold = {k: v.get("gold") or [] for k, v in qmod.questions().items()}
    out = json.loads(CTX.read_text(encoding="utf-8")) if CTX.exists() else {}
    todo = [k for k in R if k not in out]
    print("questions:", len(R), "| to build:", len(todo), flush=True)

    def one(k):
        ds = R[k]["ds"]
        v = unit(embed([R[k]["question"]])[0][:2048]).tolist()
        r = requests.post(f"http://localhost:6333/collections/{COLL[ds]}/points/search", json={"vector": v, "limit": 6, "with_payload": True}, timeout=30).json()["result"]
        top = [{"source": h["payload"]["source"], "text": h["payload"]["text"]} for h in r]
        return k, {"ds": ds, "chunks": top, "context": build_context(ds, top), "gold": gold.get(k, [])}

    with cf.ThreadPoolExecutor(6) as ex:
        for k, v in ex.map(one, todo):
            out[k] = v
    CTX.write_text(json.dumps(out), encoding="utf-8")
    print("contexts saved:", len(out))


def backend(name, slot):
    env = "NVIDIA_API_KEY"
    if name == "laguna":
        return OpenAICompatChat("poolside/laguna-xs-2.1", env, NV, extra_body={"max_tokens": 3000})
    return OpenAICompatChat("nvidia/nemotron-3.5-super-vl-120b-a12b", env, NV, extra_body={"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 3000})


def precision_per_chunk(backend, question, reference, texts):
    """Return context precision as in harness.judges.score_context_precision, with one judge call per chunk."""
    relevance = []
    for i, text in enumerate(texts, start=1):
        prompt = judges.prompts.CONTEXT_PRECISION.format(question=question, reference=reference, chunks=f"Chunk {i}:\n{text}")
        rel = judges._ask(backend, prompt).get("relevance")
        if isinstance(rel, bool):
            rel = [rel]  # with one chunk the judge sometimes returns a bare true or false
        if not isinstance(rel, list) or len(rel) != 1 or not isinstance(rel[0], bool):
            raise judges.JudgeParseError(f"one verdict expected, got {rel!r}")
        relevance.append(rel[0])
    relevant, total = 0, 0.0
    for k, ok in enumerate(relevance, start=1):
        if ok:
            relevant += 1
            total += relevant / k
    return total / relevant if relevant else 0.0


def answered():
    return {r["key"]: r["answer"] for ds in ("wix", "techqa") for r in jl(LIVEW / f"live_{ds}.jsonl") if r["status"] == "answered" and r["answer"].strip()}


def run(name):
    R = le.rows()
    ctx = json.loads(CTX.read_text(encoding="utf-8"))
    ans = answered()
    out = HERE / f"{name}.jsonl"
    kept = [r for r in jl(out) if r.get("error") is None]
    out.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
    done = {r["key"] for r in kept}
    base = name[:-2] if name.endswith("_l") else name  # a "_l" suffix means Laguna is the judge
    keys = [k for k in ctx if (k in ans if base in ("faith_nemotron", "faith_laguna", "relevancy") else True)]
    todo = [k for k in keys if k not in done]
    print(name, "to score:", len(todo), flush=True)
    judge_name = "laguna" if (name.endswith("_l") or name == "faith_laguna") else "nemotron"
    pool = [backend(judge_name, 0)]
    emb = NvEmbedder()

    gate = [threading.Lock()]
    last = [0.0]

    def one(item):
        i, k = item
        slot = 0
        b = pool[slot]
        with gate[slot]:  # pacing: at most one call per second
            wait = last[slot] + 1.0 - time.time()
            if wait > 0:
                time.sleep(wait)
            last[slot] = time.time()
        try:
            if base.startswith("faith"):
                sc = judges.score_faithfulness(b, ans[k], [ctx[k]["context"]])
            elif base == "relevancy":
                sc = judges.score_answer_relevancy(b, emb, R[k]["question"], ans[k])
            elif base == "precision":
                texts = [c["text"] for c in ctx[k]["chunks"]]
                try:
                    sc = judges.score_context_precision(b, R[k]["question"], R[k]["reference"], texts)
                except judges.JudgeParseError:
                    sc = precision_per_chunk(b, R[k]["question"], R[k]["reference"], texts)  # wrong number of verdicts: ask one chunk at a time
            else:
                sc = judges.score_context_recall(b, R[k]["reference"], [c["text"] for c in ctx[k]["chunks"]])
            return {"key": k, "score": sc, "error": None}
        except Exception as e:  # noqa: BLE001
            return {"key": k, "score": None, "error": f"{type(e).__name__}: {str(e)[:120]}"}

    with cf.ThreadPoolExecutor(10) as ex:  # request rate comes from the gate in one(); workers only overlap the waits
        for r in ex.map(one, list(enumerate(todo))):
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            time.sleep(0.3)
    rows = jl(out)
    print(name, "done: scored", sum(r.get("error") is None for r in rows), "| errors", sum(r.get("error") is not None for r in rows), flush=True)


def ci(vals, n=5000):
    rnd = random.Random(0)
    v = list(vals)
    means = sorted(statistics.fmean(v[rnd.randrange(len(v))] for _ in v) for _ in range(n))
    return statistics.fmean(v), means[int(0.025 * n)], means[int(0.975 * n)]


def report():
    ctx = json.loads(CTX.read_text(encoding="utf-8"))
    names = ("faith_nemotron", "faith_laguna", "relevancy_l", "precision_l", "recall_l")
    sc = {n: {r["key"]: r["score"] for r in jl(HERE / f"{n}.jsonl") if r.get("error") is None and r["score"] is not None} for n in names}
    fmt = lambda x: f"{x[0]:.2f} ({x[1]:.2f} to {x[2]:.2f})"  # noqa: E731
    rows = {}
    for ds, label in (("wix", "Wix"), ("techqa", "TechQA")):
        ks = [k for k in ctx if ctx[k]["ds"] == ds]
        cell = {}
        for n in names:
            v = [sc[n][k] for k in ks if k in sc[n]]
            cell[n] = (fmt(ci(v)), len(v)) if v else ("n/a", 0)
        fa = [statistics.fmean([sc[n][k] for n in ("faith_nemotron", "faith_laguna") if k in sc[n]]) for k in ks if any(k in sc[n] for n in ("faith_nemotron", "faith_laguna"))]
        cell["faith_mean"] = (fmt(ci(fa)), len(fa)) if fa else ("n/a", 0)
        gk = [k for k in ks if ctx[k]["gold"]]
        h3, h6, mrr = [], [], []
        for k in gk:
            gold = set(ctx[k]["gold"])
            arts = list(dict.fromkeys(c["source"] for c in ctx[k]["chunks"]))
            hit6 = any(c["source"] in gold for c in ctx[k]["chunks"])
            hit3 = any(a in gold for a in arts[:3])
            pos = [i + 1 for i, a in enumerate(arts) if a in gold]
            h3.append(float(hit3)); h6.append(float(hit6)); mrr.append(1 / pos[0] if pos else 0.0)
        cell["hit3"] = (fmt(ci(h3)), len(h3)) if h3 else ("n/a", 0)
        cell["hit6"] = (fmt(ci(h6)), len(h6)) if h6 else ("n/a", 0)
        cell["mrr"] = (fmt(ci(mrr)), len(mrr)) if mrr else ("n/a", 0)
        rows[label] = cell
    lines = ["| Metric | Wix | TechQA |", "|---|---|---|"]
    for key, title in (("faith_mean", "Faithfulness (mean of two judges)"), ("faith_nemotron", "  Nemotron judge"), ("faith_laguna", "  Laguna judge"), ("relevancy_l", "Answer relevancy (Laguna judge)"),
                       ("precision_l", "Context precision, 6 retrieved chunks (Laguna judge)"), ("recall_l", "Context recall, 6 retrieved chunks (Laguna judge)"), ("hit3", "Hit rate @ 3 articles"), ("hit6", "Hit rate @ 6 chunks"), ("mrr", "Mean reciprocal rank")):
        lines.append(f"| {title} | {rows['Wix'][key][0]} (n={rows['Wix'][key][1]}) | {rows['TechQA'][key][0]} (n={rows['TechQA'][key][1]}) |")
    text = "\n".join(lines)
    (HERE / "README.md").write_text("# RAG metrics of the live run (Wix and TechQA separately)\n\nBuilt by `live_metrics.py`; method in its docstring. 95% intervals from resampling the rows.\n\n" + text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    {"contexts": contexts, "run": lambda: run(sys.argv[2]), "report": report}[cmd]()
