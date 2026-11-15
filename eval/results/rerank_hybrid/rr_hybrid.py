"""Does hybrid search (keywords + meaning) or a Cohere reranker improve on the dense-only baseline? 50 Wix + 50 TechQA rows, wrong / handed-off / good rows mixed.
Base = the offline dense-only setup (Nemotron 8B + title, 2048 dims, customer question, top 6 chunks, no reranker). Same answer model (NVIDIA mistral-large-4), same prompt, same three judges.
usage (from eval/, keys loaded, COHERE_API_KEY in the environment):
  python results/rerank_hybrid/rr_hybrid.py select          picks the 100 rows (writes work/rows.json)
  python results/rerank_hybrid/rr_hybrid.py contexts        builds the contexts of the 3 arms (Cohere calls are paced and cached)
  python results/rerank_hybrid/rr_hybrid.py gen ARM         ARM = H | RR | HR  (only rows whose context differs from the base)
  python results/rerank_hybrid/rr_hybrid.py judge ARM NAME  NAME = mistral | nemotron | glm45
  python results/rerank_hybrid/rr_hybrid.py report
Arms: H = hybrid (BM25 + dense, reciprocal rank fusion k=60, top 100 each) -> top 6 chunks; RR = dense top 30 -> Cohere rerank-v3.5 -> top 6; HR = hybrid top 30 -> Cohere -> top 6.
Follow-up subcommands: contexts2 | report2 (arms B2, HT), contexts3 | report3 (arm HX); defined in their docstrings.
"""
import concurrent.futures as cf
import importlib.util
import json
import math
import os
import pathlib
import random
import re
import sys
import time
from collections import Counter, defaultdict
from math import comb

import numpy as np
import requests

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent
E = RES / "embedding_test"
WORK = HERE / "work"
WORK.mkdir(exist_ok=True)
LIVE = RES / "live_best/work"
FILES = RES.parents[1] / "infra/local_files"
JUDGES = ("mistral", "nemotron", "glm45")
ARMS = ("H", "RR", "HR")
cmd = sys.argv[1]
_argv = sys.argv[:]
sys.argv = [sys.argv[0], "none"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fr = load("full_run", RES / "model_speed/full_run.py")
qmod = load("queries", E / "queries.py")
sys.argv = _argv
a = fr.a
jl = fr.jl
_post = requests.post


def _long_post(url, *args, **kw):  # a whole-article Wix prompt can wait minutes for its reply; a 150 s timeout would discard the request
    if "nvidia" in url:
        kw["timeout"] = 420
    return _post(url, *args, **kw)


requests.post = _long_post


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


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


STOP = set("a an and are as at be by can do does for from how i if in is it my of on or that the this to was what when where which with you your".split())


def toks(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(toks(t)) for t in texts]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=np.float32)
        self.avg = self.len.mean()
        self.inv = defaultdict(list)
        for i, c in enumerate(self.tf):
            for w, n in c.items():
                self.inv[w].append((i, n))
        self.N = len(texts)

    def scores(self, q):
        s = np.zeros(self.N, dtype=np.float32)
        for w in set(toks(q)):
            post = self.inv.get(w)
            if not post:
                continue
            idf = math.log(1 + (self.N - len(post) + 0.5) / (len(post) + 0.5))
            for i, n in post:
                s[i] += idf * n * (self.k1 + 1) / (n + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def rrf(orders, k=60):
    sc = {}
    for o in orders:
        for r, i in enumerate(o):
            sc[i] = sc.get(i, 0) + 1 / (k + r + 1)
    return [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])]


def verdict_live():
    ok = {}
    for r in jl(LIVE / "live_wix.jsonl") + jl(LIVE / "live_techqa.jsonl"):
        ok[r["key"]] = r.get("status") == "answered"
    sc = {j: {r["key"]: r["score"] for r in jl(LIVE / f"judge_{j}.jsonl") if r["score"] is not None} for j in JUDGES}
    out = {}
    for k, sent in ok.items():
        if not sent:
            out[k] = "handed off"
        else:
            out[k] = "good" if sum((sc[j].get(k) or 0) >= 4 for j in JUDGES) >= 2 else "wrong"
    return out


def select():
    ctx = json.loads((E / "work/e2e_contexts.json").read_text(encoding="utf-8"))["R2"]
    live = verdict_live()
    rng = random.Random(7)
    rows = {}
    for ds, n in (("wix", 50), ("techqa", 50)):
        pool = [k for k in ctx if k.startswith(ds) and ctx[k]["set"] in ("wix_first", "wix_fresh", "techqa") and k in live]
        by = {g: [k for k in pool if live[k] == g] for g in ("wrong", "handed off", "good")}
        chosen = by["wrong"] + by["handed off"][:max(0, min(len(by["handed off"]), n // 4))]
        rest = [k for k in by["good"] if k not in chosen]
        rng.shuffle(rest)
        chosen += rest[:n - len(chosen)]
        rows.update({k: live[k] for k in chosen[:n]})
        print(ds, {g: sum(1 for k in chosen[:n] if live[k] == g) for g in by}, "available", {g: len(v) for g, v in by.items()})
    (WORK / "rows.json").write_text(json.dumps(rows), encoding="utf-8")


def cohere(query, docs, retries=6):
    for _ in range(retries):
        t0 = time.time()
        r = requests.post("https://api.cohere.com/v2/rerank", headers={"Authorization": "Bearer " + os.environ["COHERE_API_KEY"]},
                          json={"model": "rerank-v3.5", "query": query, "documents": docs, "top_n": len(docs)}, timeout=60)
        if r.status_code == 200:
            return [x["index"] for x in r.json()["results"]], round(time.time() - t0, 2)
        time.sleep(15)
    raise RuntimeError("cohere failed " + r.text[:100])


def contexts():
    qs = qmod.questions()
    rows = json.loads((WORK / "rows.json").read_text(encoding="utf-8"))
    base = json.loads((E / "work/e2e_contexts.json").read_text(encoding="utf-8"))["R2"]
    qn = json.loads((E / "work/q_nemo.json").read_text(encoding="utf-8"))
    cache_p = WORK / "cohere_cache.json"
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
    out = {arm: {} for arm in ARMS}
    for ds in ("wix", "techqa"):
        chunks = json.loads((E / f"work/chunks_{ds}.json").read_text(encoding="utf-8"))
        mat = unit(np.load(E / f"work/nemo_{ds}_title.npy")[:, :2048])
        bm = BM25([c["text"] for c in chunks])
        for k in [k for k in rows if k.startswith(ds)]:
            q = base[k]["question"]
            dense = list(np.argsort(-(mat @ unit(np.array(qn[f"{k}|question"])[:2048])))[:100])
            lex = list(np.argsort(-bm.scores(q))[:100])
            hyb = rrf([dense, lex])
            pools = {"H": hyb[:6], "RR": dense[:30], "HR": hyb[:30]}
            top_of = {}
            for arm in ARMS:
                pool = [int(i) for i in pools[arm]]
                if arm == "H":
                    order = pool
                else:
                    ck = f"{arm}|{k}"
                    if ck not in cache:
                        idx, secs = cohere(q, [chunks[i]["text"] for i in pool])
                        cache[ck] = {"order": idx, "secs": secs}
                        cache_p.write_text(json.dumps(cache), encoding="utf-8")
                        time.sleep(6.5)
                    order = [pool[j] for j in cache[ck]["order"]][:6]
                top = [chunks[i] for i in order[:6]]
                top_of[arm] = top
                ctx = build_context(ds, top)
                gold = qs[k]["gold"]
                arts3 = list(dict.fromkeys(c["source"] for c in top))[:3]
                out[arm][k] = {"set": base[k]["set"], "question": q, "reference": base[k]["reference"], "context": ctx, "system": base[k]["system"],
                               "gold_in_context": any(g in ctx for g in gold), "art3": any(s in gold for s in arts3), "changed": ctx != base[k]["context"]}
            print(k, {arm: (out[arm][k]["gold_in_context"], out[arm][k]["changed"]) for arm in ARMS}, flush=True)
    (WORK / "arm_contexts.json").write_text(json.dumps(out), encoding="utf-8")
    base_ok = sum(1 for k in rows if any(g in base[k]["context"] for g in qs[k]["gold"]))
    print("base gold in context:", base_ok, "of", len(rows), "| changed contexts:", {arm: sum(v["changed"] for v in out[arm].values()) for arm in ARMS})


def contexts2():
    """Add arms B2 and HT to arm_contexts.json. B2 = the dense baseline run a second time on the same 100 rows (noise control); HT = hybrid + Cohere rerank on the other 50 of the original 100 TechQA rows."""
    qs = qmod.questions()
    rows = json.loads((WORK / "rows.json").read_text(encoding="utf-8"))
    base = json.loads((E / "work/e2e_contexts.json").read_text(encoding="utf-8"))["R2"]
    qn = json.loads((E / "work/q_nemo.json").read_text(encoding="utf-8"))
    actx = json.loads((WORK / "arm_contexts.json").read_text(encoding="utf-8"))
    cache_p = WORK / "cohere_cache.json"
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
    actx["B2"] = {k: {**{f: base[k][f] for f in ("set", "question", "reference", "context", "system")}, "gold_in_context": any(g in base[k]["context"] for g in qs[k]["gold"]), "art3": True, "changed": True} for k in rows}
    chunks = json.loads((E / "work/chunks_techqa.json").read_text(encoding="utf-8"))
    mat = unit(np.load(E / "work/nemo_techqa_title.npy")[:, :2048])
    bm = BM25([c["text"] for c in chunks])
    actx["HT"] = {}
    for k in [k for k in base if k.startswith("techqa") and k not in rows]:
        q = base[k]["question"]
        dense = list(np.argsort(-(mat @ unit(np.array(qn[f"{k}|question"])[:2048])))[:100])
        hyb = [int(i) for i in rrf([dense, list(np.argsort(-bm.scores(q))[:100])])[:30]]
        ck = f"HR|{k}"
        if ck not in cache:
            idx, secs = cohere(q, [chunks[i]["text"] for i in hyb])
            cache[ck] = {"order": idx, "secs": secs}
            cache_p.write_text(json.dumps(cache), encoding="utf-8")
            time.sleep(6.5)
        top = [chunks[hyb[j]] for j in cache[ck]["order"]][:6]
        ctx = build_context("techqa", top)
        actx["HT"][k] = {"set": base[k]["set"], "question": q, "reference": base[k]["reference"], "context": ctx, "system": base[k]["system"],
                         "gold_in_context": any(g in ctx for g in qs[k]["gold"]), "art3": any(c["source"] in qs[k]["gold"] for c in top[:3]), "changed": True}
        print(k, flush=True)
    (WORK / "arm_contexts.json").write_text(json.dumps(actx), encoding="utf-8")
    print("arms added: B2", len(actx["B2"]), "HT", len(actx["HT"]))


def contexts3():
    """Add arm HX to arm_contexts.json: hybrid alone (BM25 + dense, reciprocal rank fusion, top 6 chunks, no Cohere) on the 150 Wix rows outside the first 100-row test."""
    qs = qmod.questions()
    rows = json.loads((WORK / "rows.json").read_text(encoding="utf-8"))
    base = json.loads((E / "work/e2e_contexts.json").read_text(encoding="utf-8"))["R2"]
    qn = json.loads((E / "work/q_nemo.json").read_text(encoding="utf-8"))
    actx = json.loads((WORK / "arm_contexts.json").read_text(encoding="utf-8"))
    chunks = json.loads((E / "work/chunks_wix.json").read_text(encoding="utf-8"))
    mat = unit(np.load(E / "work/nemo_wix_title.npy")[:, :2048])
    bm = BM25([c["text"] for c in chunks])
    actx["HX"] = {}
    for k in [k for k in base if k.startswith("wix") and k not in rows]:
        q = base[k]["question"]
        dense = list(np.argsort(-(mat @ unit(np.array(qn[f"{k}|question"])[:2048])))[:100])
        top = [chunks[int(i)] for i in rrf([dense, list(np.argsort(-bm.scores(q))[:100])])[:6]]
        ctx = build_context("wix", top)
        actx["HX"][k] = {"set": base[k]["set"], "question": q, "reference": base[k]["reference"], "context": ctx, "system": base[k]["system"],
                         "gold_in_context": any(g in ctx for g in qs[k]["gold"]), "art3": any(c["source"] in qs[k]["gold"] for c in top[:3]), "changed": True}
    (WORK / "arm_contexts.json").write_text(json.dumps(actx), encoding="utf-8")
    print("arm added: HX", len(actx["HX"]), "| context differs from the current setup in", sum(v["context"] != base[k]["context"] for k, v in actx["HX"].items()), "rows")


def report3():
    """Compare hybrid alone with the dense baseline on all 200 Wix rows (50 from the first test, 150 new), also split by the live outcome."""
    rows = json.loads((WORK / "rows.json").read_text(encoding="utf-8"))
    live = verdict_live()
    base_g = {r["key"]: r for r in jl(E / "work/e2e_gen_R2.jsonl") if r.get("tokens")}
    base_s = {j: {r["key"]: r["score"] for r in jl(E / f"work/e2e_judge_{j}_R2.jsonl") if r["score"] is not None} for j in JUDGES}

    def data(arm):
        return ({r["key"]: r for r in jl(WORK / f"gen_{arm}.jsonl") if r.get("tokens")},
                {j: {r["key"]: r["score"] for r in jl(WORK / f"judge_{j}_{arm}.jsonl") + jl(WORK / f"judge_{j}_{arm}.s1.jsonl") if r["score"] is not None} for j in JUDGES})

    def outcome(g, s, k):
        if k not in g or not g[k]["sent"]:
            return "ho"
        return "good" if sum((s[j].get(k) or 0) >= 4 for j in JUDGES) >= 2 else "wrong"

    hx, h = data("HX"), data("H")
    keys = [k for k in base_g if k.startswith("wix")]
    ob = {k: outcome(base_g, base_s, k) for k in keys}
    oh = {k: outcome(*h, k) if k in rows else outcome(*hx, k) for k in keys}
    new = [k for k in keys if k not in rows]
    miss = sum(1 for k in new if k in hx[0] and hx[0][k]["sent"] and sum(k in hx[1][j] for j in JUDGES) < 3)

    def line(name, vs):
        c = Counter(vs.values())
        return f"{name:34s} | {len(vs)} | {c['good']} | {c['wrong']} | {c['ho']} | {100 * c['wrong'] / max(1, c['good'] + c['wrong']):.1f}% | {(c['good'] - c['wrong']) / len(vs):+.3f}"

    print("setup | rows | good | wrong | handed off | wrong share of sent | score")
    for label, ks in (("ALL 200 Wix rows", keys), ("the 150 NEW rows", new), ("the 50 rows of the first test", [k for k in keys if k in rows])):
        print(f"-- {label}")
        print(line("current setup (dense only)", {k: ob[k] for k in ks}))
        print(line("hybrid alone", {k: oh[k] for k in ks}))
    mv = Counter((ob[k], oh[k]) for k in keys if ob[k] != oh[k])
    print("rows that changed outcome (current -> hybrid):", dict(mv))
    gained = sum(1 for k in keys if oh[k] == "good" and ob[k] != "good"); lost = sum(1 for k in keys if ob[k] == "good" and oh[k] != "good")
    n = gained + lost
    print(f"good answers gained {gained}, lost {lost}; exact two-sided p = {min(1.0, 2 * sum(comb(n, i) for i in range(min(gained, lost) + 1)) / 2 ** n) if n else 1.0:.3f}")
    wf = sum(1 for k in keys if ob[k] == "wrong" and oh[k] != "wrong"); wa = sum(1 for k in keys if ob[k] != "wrong" and oh[k] == "wrong")
    print(f"wrong answers removed {wf}, added {wa}; missing grades in new rows: {miss}")
    for grp in ("good", "wrong", "handed off"):
        kk = [k for k in keys if live.get(k) == grp]
        print(f"rows the live run had as {grp} ({len(kk)}): current {dict(Counter(ob[k] for k in kk))} -> hybrid {dict(Counter(oh[k] for k in kk))}")


def gen(arm):
    ctx = json.loads((WORK / "arm_contexts.json").read_text(encoding="utf-8"))[arm]
    out = WORK / f"gen_{arm}.jsonl"
    rows = jl(out)
    done = {r["key"] for r in rows if r.get("tokens")}
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r.get("tokens")), encoding="utf-8")
    todo = [(k, c) for k, c in ctx.items() if c["changed"] and k not in done]
    print("arm", arm, "to answer", len(todo), flush=True)
    with cf.ThreadPoolExecutor(int(os.environ.get("GEN_WORKERS", "3"))) as ex:
        for r in ex.map(lambda kc: fr.one("ML4", kc[0], kc[1]), todo):
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("gen", arm, "done; without answer:", [r["key"] for r in jl(out) if not r.get("tokens")], flush=True)


def judge(arm, name):
    b = a.backend(name)
    ctx = json.loads((WORK / "arm_contexts.json").read_text(encoding="utf-8"))[arm]
    shard = os.environ.get("JUDGE_SHARD")  # "1": second shard (own output file, odd positions); unset: first shard (even positions when JUDGE_SPLIT is set, else all)
    out = WORK / f"judge_{name}_{arm}{'.s1' if shard == '1' else ''}.jsonl"
    other = WORK / f"judge_{name}_{arm}{'' if shard == '1' else '.s1'}.jsonl"
    gens = {r["key"]: r for r in jl(WORK / f"gen_{arm}.jsonl") if r.get("tokens") and r["sent"] and r["answer"].strip()}
    for _ in range(4):
        kept = [r for r in jl(out) if r["score"] is not None and r["key"] in gens]
        out.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
        done = {r["key"] for r in kept} | {r["key"] for r in jl(other) if r["score"] is not None}
        order = sorted(gens)
        mine = [k for i, k in enumerate(order) if not os.environ.get("JUDGE_SPLIT") or i % 2 == (1 if shard == "1" else 0)]
        todo = [k for k in mine if k not in done]
        if not todo:
            break
        for k in todo:
            try:
                sc, err = fr.score_answer_correctness(b, ctx[k]["question"], ctx[k]["reference"], gens[k]["answer"]), None
            except Exception as e:  # noqa: BLE001
                sc, err = None, type(e).__name__
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": k, "score": sc, "error": err}) + "\n")
            time.sleep({"mistral": 1.2, "nemotron": 0.4}.get(name, 0.8))
        time.sleep(20)
    print("judge", name, arm, "complete", len([r for r in jl(out) if r["score"] is not None]), "of", len(gens), flush=True)


def report():
    rows = json.loads((WORK / "rows.json").read_text(encoding="utf-8"))
    actx = json.loads((WORK / "arm_contexts.json").read_text(encoding="utf-8"))
    base_g = {r["key"]: r for r in jl(E / "work/e2e_gen_R2.jsonl") if r.get("tokens")}
    base_s = {j: {r["key"]: r["score"] for r in jl(E / f"work/e2e_judge_{j}_R2.jsonl") if r["score"] is not None} for j in JUDGES}
    G, S = {"base": base_g}, {"base": base_s}
    for arm in ARMS:
        g = dict(base_g)
        g.update({r["key"]: r for r in jl(WORK / f"gen_{arm}.jsonl") if r.get("tokens")})
        s = {j: dict(base_s[j]) for j in JUDGES}
        for j in JUDGES:
            mine = jl(WORK / f"judge_{j}_{arm}.jsonl") + jl(WORK / f"judge_{j}_{arm}.s1.jsonl")
            s[j].update({r["key"]: r["score"] for r in mine if r["score"] is not None})
            for k, c in actx[arm].items():  # a changed row keeps only its new grades
                if c["changed"] and k not in {r["key"] for r in mine}:
                    s[j].pop(k, None)
        G[arm], S[arm] = g, s
    good = lambda arm, k: sum((S[arm][j].get(k) or 0) >= 4 for j in JUDGES) >= 2  # noqa: E731
    names = {"base": "Base (dense only, no reranker) = current best", "H": "H  hybrid (BM25 + dense)", "RR": "RR dense + Cohere rerank", "HR": "HR hybrid + Cohere rerank"}
    for label, ds in (("Wix 50", "wix"), ("TechQA 50", "techqa")):
        ks = [k for k in rows if k.startswith(ds)]
        print(f"\n=== {label}: live verdicts of these rows {dict(Counter(rows[k] for k in ks))} ===")
        print("arm | gold in context | gold article in top 3 | rows with a changed context | good | wrong | handed off | score | missing grades")
        for arm in ("base",) + ARMS:
            def v(k):
                if k not in G[arm] or not G[arm][k]["sent"]:
                    return "ho"
                return "good" if good(arm, k) else "wrong"
            vs = {k: v(k) for k in ks}
            c = Counter(vs.values())
            if arm == "base":
                gic = sum(any(g in json.loads((E / 'work/e2e_contexts.json').read_text(encoding='utf-8'))['R2'][k]['context'] for g in qmod.questions()[k]['gold']) for k in ks)
                a3, ch = "-", "-"
            else:
                gic = sum(actx[arm][k]["gold_in_context"] for k in ks)
                a3 = sum(actx[arm][k]["art3"] for k in ks)
                ch = sum(actx[arm][k]["changed"] for k in ks)
            miss = sum(1 for k in ks if vs[k] != "ho" and sum(k in S[arm][j] for j in JUDGES) < 3)
            print(f"{names[arm]:46s} | {gic} | {a3} | {ch} | {c['good']} | {c['wrong']} | {c['ho']} | {(c['good'] - c['wrong']) / len(ks):+.2f} | {miss}")
        for arm in ARMS:
            vb = {k: ("ho" if not (k in G['base'] and G['base'][k]['sent']) else ("good" if good('base', k) else "wrong")) for k in ks}
            va = {k: ("ho" if not (k in G[arm] and G[arm][k]['sent']) else ("good" if good(arm, k) else "wrong")) for k in ks}
            moves = Counter((vb[k], va[k]) for k in ks if vb[k] != va[k])
            print(f"  {arm} vs base, rows that changed outcome:", dict(moves) or "none")
            for grp in ("wrong", "handed off", "good"):
                kk = [k for k in ks if rows[k] == grp]
                print(f"     live-{grp} rows ({len(kk)}): base good {sum(vb[k] == 'good' for k in kk)} wrong {sum(vb[k] == 'wrong' for k in kk)} ho {sum(vb[k] == 'ho' for k in kk)} -> {arm} good {sum(va[k] == 'good' for k in kk)} wrong {sum(va[k] == 'wrong' for k in kk)} ho {sum(va[k] == 'ho' for k in kk)}")
    cache = json.loads((WORK / "cohere_cache.json").read_text(encoding="utf-8")) if (WORK / "cohere_cache.json").exists() else {}
    secs = [v["secs"] for v in cache.values()]
    if secs:
        print(f"\nCohere rerank call: median {np.median(secs):.2f}s, 90th pct {np.percentile(secs, 90):.2f}s, {len(secs)} calls")


def report2():
    """Report (A) the noise control, the dense baseline run twice on the same 100 rows, and (B) hybrid + rerank vs the dense baseline on all 100 original TechQA rows."""
    rows = json.loads((WORK / "rows.json").read_text(encoding="utf-8"))
    live = verdict_live()
    base_g = {r["key"]: r for r in jl(E / "work/e2e_gen_R2.jsonl") if r.get("tokens")}
    base_s = {j: {r["key"]: r["score"] for r in jl(E / f"work/e2e_judge_{j}_R2.jsonl") if r["score"] is not None} for j in JUDGES}

    def arm_data(arm):
        g = {r["key"]: r for r in jl(WORK / f"gen_{arm}.jsonl") if r.get("tokens")}
        s = {j: {r["key"]: r["score"] for r in jl(WORK / f"judge_{j}_{arm}.jsonl") + jl(WORK / f"judge_{j}_{arm}.s1.jsonl") if r["score"] is not None} for j in JUDGES}
        return g, s

    def outcome(g, s, k):
        if k not in g or not g[k]["sent"]:
            return "ho"
        return "good" if sum((s[j].get(k) or 0) >= 4 for j in JUDGES) >= 2 else "wrong"

    def missing(g, s, ks):
        return sum(1 for k in ks if k in g and g[k]["sent"] and sum(k in s[j] for j in JUDGES) < 3)

    def line(name, vs):
        c = Counter(vs.values())
        n = len(vs)
        return f"{name:44s} | {n} | {c['good']} | {c['wrong']} | {c['ho']} | {100 * c['wrong'] / max(1, c['good'] + c['wrong']):.1f}% | {(c['good'] - c['wrong']) / n:+.3f}"

    arms = {a: arm_data(a) for a in ("B2", "H", "RR", "HR", "HT")}
    ob = {k: outcome(base_g, base_s, k) for k in rows}
    print("=== A. Noise control: the current setup, run twice on the same 100 rows (same contexts, temperature 0) ===")
    print("run | rows | good | wrong | handed off | wrong share of sent | score")
    for label, ds in (("Wix 50", "wix"), ("TechQA 50", "techqa"), ("All 100", "")):
        ks = [k for k in rows if k.startswith(ds)]
        o2 = {k: outcome(*arms["B2"], k) for k in ks}
        print(f"-- {label}")
        print(line("current setup, first run", {k: ob[k] for k in ks}))
        print(line("current setup, second run (B2)", o2))
        diff = Counter((ob[k], o2[k]) for k in ks if ob[k] != o2[k])
        print(f"   rows whose outcome changed between the two runs: {sum(diff.values())} of {len(ks)} {dict(diff)}; missing grades in B2: {missing(*arms['B2'], ks)}")
        for a in ("H", "RR", "HR"):
            oa = {k: outcome(*arms[a], k) for k in ks}
            print(line(f"{a} (one run)", oa))
            print(f"   {a} vs first run: score {((Counter(oa.values())['good'] - Counter(oa.values())['wrong']) - (Counter(ob[k] for k in ks)['good'] - Counter(ob[k] for k in ks)['wrong'])) / len(ks):+.3f}; "
                  f"vs second run: {((Counter(oa.values())['good'] - Counter(oa.values())['wrong']) - (Counter(o2.values())['good'] - Counter(o2.values())['wrong'])) / len(ks):+.3f}")
    print("\n=== B. Hybrid + Cohere rerank (HR) vs the current setup on all 100 original TechQA rows (50 from the first run, 50 new) ===")
    base_all = {k: outcome(base_g, base_s, k) for k in base_g if k.startswith("techqa")}
    hr_all = {}
    for k in base_all:
        hr_all[k] = outcome(*arms["HR"], k) if k in rows else outcome(*arms["HT"], k)
    print("setup | rows | good | wrong | handed off | wrong share of sent | score")
    print(line("current setup (dense only)", base_all))
    print(line("hybrid + rerank (HR)", hr_all))
    mv = Counter((base_all[k], hr_all[k]) for k in base_all if base_all[k] != hr_all[k])
    print("   rows that changed outcome:", dict(mv))
    new = [k for k in base_all if k not in rows]
    print(f"   on the {len(new)} NEW rows only: current {Counter(base_all[k] for k in new)} -> HR {Counter(hr_all[k] for k in new)}")
    gained = sum(1 for k in base_all if hr_all[k] == "good" and base_all[k] != "good")
    lost = sum(1 for k in base_all if base_all[k] == "good" and hr_all[k] != "good")
    print(f"   good answers gained {gained}, lost {lost}; exact two-sided p = {min(1.0, 2 * sum(comb(gained + lost, i) for i in range(min(gained, lost) + 1)) / 2 ** (gained + lost)) if gained + lost else 1.0:.3f}")
    wr_f = sum(1 for k in base_all if base_all[k] == "wrong" and hr_all[k] != "wrong"); wr_a = sum(1 for k in base_all if base_all[k] != "wrong" and hr_all[k] == "wrong")
    print(f"   wrong answers removed {wr_f}, added {wr_a}")
    print(f"   missing grades: HT {missing(*arms['HT'], new)}")
    for grp in ("good", "wrong", "handed off"):
        kk = [k for k in base_all if live[k] == grp]
        print(f"   rows the live run had as {grp} ({len(kk)}): current {dict(Counter(base_all[k] for k in kk))} -> HR {dict(Counter(hr_all[k] for k in kk))}")
    secs = [v["secs"] for v in (json.loads((WORK / "cohere_cache.json").read_text(encoding="utf-8")) if (WORK / "cohere_cache.json").exists() else {}).values()]
    if secs:
        print(f"   Cohere calls: {len(secs)}, median {np.median(secs):.2f}s")


if __name__ == "__main__":
    {"select": select, "contexts": contexts, "contexts2": contexts2, "report2": report2, "contexts3": contexts3, "report3": report3, "gen": lambda: gen(sys.argv[2]), "judge": lambda: judge(sys.argv[2], sys.argv[3]), "report": report}[cmd]()
