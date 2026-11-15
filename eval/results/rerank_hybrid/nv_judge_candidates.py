"""Candidate judges from other model families, hosted on NVIDIA, tested on the 60 answers of judge_candidates.py (every panel-wrong answer plus good ones, seed 3).
Also times the panel judges nemotron and glm45 under the same conditions (4 requests at once, same prompt).
usage (from eval/, keys loaded): python results/rerank_hybrid/nv_judge_candidates.py NAME   (NAME in CANDS, or 'nemotron' / 'glm45' for the timing run)   |   ... report
Results: work/nv_{NAME}.jsonl (key, score, secs, err); 'report' prints the comparison table."""
import concurrent.futures as cf
import json
import os
import pathlib
import random
import re
import sys
import time

import requests

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness import nvidia_key, prompts  # noqa: E402

E = HERE.parent / "embedding_test/work"
J = ("mistral", "nemotron", "glm45")
NV = "https://integrate.api.nvidia.com/v1/chat/completions"
OFF = {"chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
CANDS = {
    "gemma4": ("google/gemma-4-31b-it", {}),
    "step5": ("stepfun-ai/step-5-preview", {}),
    "laguna": ("poolside/laguna-xs-2.1", {}),
    "nemotron": ("nvidia/nemotron-3.5-super-vl-120b-a12b", {"chat_template_kwargs": {"enable_thinking": False}}),
}
ALL = list(CANDS) + ["glm45"]


def jl(p):
    return [json.loads(x) for x in pathlib.Path(p).read_text(encoding="utf-8").splitlines() if x.strip()] if pathlib.Path(p).exists() else []


ctx = json.loads((E / "e2e_contexts.json").read_text(encoding="utf-8"))["R2"]
gen = {r["key"]: r for r in jl(E / "e2e_gen_R2.jsonl") if r.get("tokens") and r["sent"] and r["answer"].strip()}
sc = {j: {r["key"]: r["score"] for r in jl(E / f"e2e_judge_{j}_R2.jsonl") if r["score"] is not None} for j in J}
keys = [k for k in gen if all(k in sc[j] for j in J)]
panel = {k: sum(sc[j][k] >= 4 for j in J) >= 2 for k in keys}
bad = [k for k in keys if not panel[k]]
good = [k for k in keys if panel[k]]
random.Random(3).shuffle(good)
rows = bad + good[:60 - len(bad)]


def call(name, k):
    prompt = prompts.ANSWER_CORRECTNESS.format(reference=ctx[k]["reference"], answer=gen[k]["answer"])
    if name == "glm45":
        url, key, body = "https://api.z.ai/api/paas/v4/chat/completions", os.environ["ZAI_API_KEY"], {"model": "glm-4.5-flash", "thinking": {"type": "disabled"}, "max_tokens": 400}
    else:
        model, extra = CANDS[name]
        url, key, body = NV, nvidia_key.api_key(), {"model": model, "max_tokens": 400 if name == "nemotron" else 2500, **extra}
    body.update({"temperature": 0, "messages": [{"role": "user", "content": prompt}]})
    score, err, t0 = None, None, time.time()
    for attempt in range(4):
        try:
            r = requests.post(url, headers={"Authorization": "Bearer " + key}, json=body, timeout=180)
        except Exception as e:  # noqa: BLE001
            err = type(e).__name__
            time.sleep(5)
            continue
        if r.status_code == 200:
            c = r.json()["choices"][0]["message"].get("content")
            if isinstance(c, list):
                c = "".join(b.get("text", "") for b in c if isinstance(b, dict))
            m = re.search(r"\{.*\}", c or "", re.S)
            try:
                v = json.loads(m.group(0))["score"]
                score, err = (int(v) if v == int(v) and 1 <= v <= 5 else None), None
            except (ValueError, KeyError, TypeError, AttributeError):
                err = "parse"
            break
        err = f"http{r.status_code}"
        if r.status_code == 400:
            break
        time.sleep(15 if r.status_code == 429 else 5)
    return {"key": k, "score": score, "secs": round(time.time() - t0, 1), "err": err}


def run(name):
    out = HERE / f"work/nv_{name}.jsonl"
    done = {r["key"] for r in jl(out) if r["score"] is not None}
    todo = [k for k in rows if k not in done]
    t0 = time.time()
    with cf.ThreadPoolExecutor(4) as ex:
        for r in ex.map(lambda k: call(name, k), todo):
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
    wall = time.time() - t0
    (HERE / f"work/nv_{name}.wall").write_text(str(round(wall, 1)))
    print(name, "done", len(todo), "calls in", round(wall), "s", flush=True)


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def report():
    gm = {j: {k: sc[j][k] >= 4 for k in rows} for j in J}
    print(f"60 answers: {len(bad)} that the panel calls wrong + {len(rows) - len(bad)} it calls good (hard on purpose: judges disagree most here)")
    print("judge | graded | agrees with others' majority | agrees with ministral-14b / nemotron / glm45 | kappa vs majority | median s/call | rows/min (4 at once) | 400 rows takes")
    for name in ALL:
        res = {r["key"]: r for r in jl(HERE / f"work/nv_{name}.jsonl") if r["score"] is not None}
        if len(res) < 5:
            continue
        ks = [k for k in rows if k in res]
        g = {k: res[k]["score"] >= 4 for k in ks}
        pm = {k: panel[k] for k in ks}
        secs = sorted(r["secs"] for r in res.values())
        wp = HERE / f"work/nv_{name}.wall"
        wall = float(wp.read_text()) if wp.exists() else None
        rpm = f"{60 * len(ks) / wall:.1f}" if wall else "?"
        eta = f"{400 / (60 * len(ks) / wall):.0f} min" if wall else "?"
        pair = " / ".join(f"{sum(g[k] == gm[x][k] for k in ks)}/{len(ks)}" for x in J)
        fp = sum(g[k] and not pm[k] for k in ks); tn = sum((not g[k]) and not pm[k] for k in ks)
        fn = sum((not g[k]) and pm[k] for k in ks); tp = sum(g[k] and pm[k] for k in ks)
        print(f"{name} | {len(ks)} | agrees with panel {sum(g[k] == pm[k] for k in ks)}/{len(ks)} | vs 14b/nemo/glm {pair} | kappa {kappa([g[k] for k in ks], [pm[k] for k in ks]):.2f} | "
              f"calls panel-wrong answers good: {fp} of {fp + tn}; panel-good answers wrong: {fn} of {fn + tp} | median {secs[len(secs) // 2]}s | {rpm} rows/min | 400 rows: {eta} | errors {sum(1 for r in jl(HERE / f'work/nv_{name}.jsonl') if r['score'] is None)}")
    print("-- panel judges, from their saved scores (leave-one-out: the judge vs the majority of the OTHER two, ties count as wrong)")
    for j in J:
        a, b = [x for x in J if x != j]
        loo = {k: gm[a][k] and gm[b][k] for k in rows}
        print(f"{j} | agrees with other two both saying good {sum(gm[j][k] == loo[k] for k in rows)}/{len(rows)} | vs ministral/nemotron/glm {' / '.join(str(sum(gm[j][k] == gm[x][k] for k in rows)) for x in J)}/{len(rows)} | gives good to {sum(gm[j].values())} of {len(rows)}")


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
    else:
        run(sys.argv[1])
