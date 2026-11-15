"""Full run of the candidate judge Laguna (Poolside, via NVIDIA) on every answer the three panel judges (ministral-14b, nemotron, glm45) have graded: the 300 base answers (R2) and all answers of the
hybrid / reranker / noise-control arms. Same prompt as the panel (reference + answer, 1-5 score), temperature 0.
Compares Laguna with each panel judge and recomputes the headline numbers with the panel (nemotron, glm45, laguna) in place of (ministral-14b, nemotron, glm45).
usage (from eval/, keys loaded): python results/rerank_hybrid/laguna_full.py run     |     ... report"""
import concurrent.futures as cf
import json
import pathlib
import re
import sys
import time

import requests

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness import nvidia_key, prompts  # noqa: E402

W = HERE / "work"
E = HERE.parent / "embedding_test/work"
J = ("mistral", "nemotron", "glm45")
ARMS = ("H", "RR", "HR", "B2", "HT", "HX")
OUT = W / "laguna_full.jsonl"


def jl(p):
    return [json.loads(x) for x in pathlib.Path(p).read_text(encoding="utf-8").splitlines() if x.strip()] if pathlib.Path(p).exists() else []


def collect():
    """unit id ("ARM|key") -> {src, key, ref, answer, sent, scores{judge: int}}"""
    units = {}
    base_ctx = json.loads((E / "e2e_contexts.json").read_text(encoding="utf-8"))["R2"]
    arm_ctx = json.loads((W / "arm_contexts.json").read_text(encoding="utf-8"))
    gens = {"R2": {r["key"]: r for r in jl(E / "e2e_gen_R2.jsonl") if r.get("tokens")}}
    scs = {"R2": {j: {r["key"]: r["score"] for r in jl(E / f"e2e_judge_{j}_R2.jsonl") if r["score"] is not None} for j in J}}
    refs = {"R2": {k: v["reference"] for k, v in base_ctx.items()}}
    for a in ARMS:
        gens[a] = {r["key"]: r for r in jl(W / f"gen_{a}.jsonl") if r.get("tokens")}
        scs[a] = {j: {r["key"]: r["score"] for r in jl(W / f"judge_{j}_{a}.jsonl") + jl(W / f"judge_{j}_{a}.s1.jsonl") if r["score"] is not None} for j in J}
        refs[a] = {k: v["reference"] for k, v in arm_ctx[a].items()}
    for src in gens:
        for k, g in gens[src].items():
            u = f"{src}|{k}"
            sent = bool(g["sent"] and g["answer"].strip())
            units[u] = {"src": src, "key": k, "ref": refs[src].get(k), "answer": g["answer"], "sent": sent, "scores": {j: scs[src][j].get(k) for j in J}}
    return units


def ask(u, d):
    prompt = prompts.ANSWER_CORRECTNESS.format(reference=d["ref"], answer=d["answer"])
    body = {"model": "poolside/laguna-xs-2.1", "temperature": 0, "max_tokens": 2500, "messages": [{"role": "user", "content": prompt}]}
    score, err, t0 = None, None, time.time()
    for _ in range(6):
        try:
            r = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers={"Authorization": "Bearer " + nvidia_key.api_key()}, json=body, timeout=120)
        except Exception as e:  # noqa: BLE001
            err = type(e).__name__
            time.sleep(5)
            continue
        if r.status_code == 200:
            c = r.json()["choices"][0]["message"].get("content") or ""
            m = re.search(r"\{.*\}", c, re.S)
            try:
                v = json.loads(m.group(0))["score"]
                score, err = (int(v) if v == int(v) and 1 <= v <= 5 else None), None
            except (ValueError, KeyError, TypeError, AttributeError):
                err = "parse"
            break
        err = f"http{r.status_code}"
        time.sleep(15 if r.status_code == 429 else 5)
    return {"unit": u, "score": score, "secs": round(time.time() - t0, 2), "err": err}


def run():
    units = {u: d for u, d in collect().items() if d["sent"] and d["ref"] and all(d["scores"][j] is not None for j in J)}
    done = {r["unit"] for r in jl(OUT) if r["score"] is not None}
    OUT.write_text("".join(json.dumps(r) + "\n" for r in jl(OUT) if r["score"] is not None), encoding="utf-8")
    todo = [u for u in units if u not in done]
    print("answers graded by all three panel judges:", len(units), "| to grade with Laguna:", len(todo), flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(6) as ex:
        for i, r in enumerate(ex.map(lambda u: ask(u, units[u]), todo), 1):
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            if i % 100 == 0:
                print(i, "done", round(time.time() - t0), "s", flush=True)
    wall = time.time() - t0
    print(f"graded {len(todo)} in {wall:.0f} s = {60 * len(todo) / max(wall, 1):.0f} rows/min", flush=True)


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def report():
    units = collect()
    lag = {r["unit"]: r["score"] for r in jl(OUT) if r["score"] is not None}
    rows = [u for u in lag if u in units and units[u]["sent"]]
    n = len(rows)
    S = {j: {u: units[u]["scores"][j] for u in rows} for j in J}
    S["laguna"] = {u: lag[u] for u in rows}
    names = ("mistral", "nemotron", "glm45", "laguna")
    g = {j: {u: S[j][u] >= 4 for u in rows} for j in names}
    print(f"== {n} answers graded by all four judges (base 300 + hybrid/reranker arms)")
    print("-- how often each judge says good (score >= 4):", {j: f"{sum(g[j].values())} ({100 * sum(g[j].values()) / n:.0f}%)" for j in names})
    print("-- good / not-good agreement between two judges (count of answers, percent, kappa)")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ag = sum(g[a][u] == g[b][u] for u in rows)
            print(f"   {a:8s} vs {b:8s} {ag}/{n} = {100 * ag / n:.1f}%  kappa {kappa([g[a][u] for u in rows], [g[b][u] for u in rows]):.2f}  exact same 1-5 score {sum(S[a][u] == S[b][u] for u in rows)}")
    print("-- each judge vs the majority of the OTHER THREE (needs 2 of 3), fair comparison, same rows")
    for j in names:
        others = [x for x in names if x != j]
        maj = {u: sum(g[x][u] for x in others) >= 2 for u in rows}
        ag = sum(g[j][u] == maj[u] for u in rows)
        lo, hi = wilson(ag, n)
        miss = sum(g[j][u] and not maj[u] for u in rows); harsh = sum((not g[j][u]) and maj[u] for u in rows)
        print(f"   {j:8s} {ag}/{n} = {100 * ag / n:.1f}% (95% range {100 * lo:.1f}-{100 * hi:.1f}) | says good where the others say not good: {miss} | says not good where the others say good: {harsh}")
    old = {u: sum(g[j][u] for j in J) >= 2 for u in rows}
    new = {u: sum(g[j][u] for j in ("nemotron", "glm45", "laguna")) >= 2 for u in rows}
    flips = sum(old[u] != new[u] for u in rows)
    print(f"-- panel old (ministral, nemotron, glm45) vs new (nemotron, glm45, laguna): same verdict on {n - flips}/{n}, differ on {flips} (old good -> new not good {sum(old[u] and not new[u] for u in rows)}, old not good -> new good {sum(new[u] and not old[u] for u in rows)})")
    print("-- headline of the live base answers (R2: 200 Wix + 100 TechQA incl. handed off), old panel vs new panel")
    for ds in ("wix", "techqa"):
        ks = [k for k, d in units.items() if d["src"] == "R2" and d["key"].startswith(ds)]
        tot = len(ks)
        og = sum(old.get(k, False) for k in ks if k in lag); ow = sum((not old[k]) for k in ks if k in lag)
        ng = sum(new.get(k, False) for k in ks if k in lag); nw = sum((not new[k]) for k in ks if k in lag)
        print(f"   {ds}: rows {tot} | old panel good {og} wrong {ow} score {(og - ow) / tot:+.3f} | new panel good {ng} wrong {nw} score {(ng - nw) / tot:+.3f}")
    print("-- arms (answers that were generated for each arm): good sent / wrong sent, old panel -> new panel")
    for a in ARMS:
        ks = [u for u in rows if units[u]["src"] == a]
        if ks:
            print(f"   {a:3s} {len(ks)} sent | old {sum(old[u] for u in ks)}/{sum(not old[u] for u in ks)} | new {sum(new[u] for u in ks)}/{sum(not new[u] for u in ks)}")
    dis = [u for u in rows if g["laguna"][u] != (sum(g[x][u] for x in ("nemotron", "glm45")) == 2)]
    print(f"-- Laguna differs from 'nemotron AND glm45 agree': {len(dis)} answers. Score pattern (mistral/nemotron/glm45 -> laguna), first 12:")
    for u in dis[:12]:
        print("  ", u, [S[j][u] for j in J], "->", S["laguna"][u])
    secs = sorted(r["secs"] for r in jl(OUT) if r.get("secs") is not None)
    print(f"-- Laguna median {secs[len(secs) // 2]} s per call; errors in file: {sum(1 for r in jl(OUT) if r['score'] is None)}")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
