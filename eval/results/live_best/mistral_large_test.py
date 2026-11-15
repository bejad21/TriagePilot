"""Does Mistral's own hosted mistral-large-4 answer as well as the NVIDIA-hosted mistral-large-4-1t-a52b? Same 200 rows (the Wix tuning set, the first 100 TechQA questions), same saved contexts (the best version's search), same prompt, temperature 0.
usage (from eval/, MISTRAL_API_KEY and the judge keys loaded):
  python results/live_best/mistral_large_test.py gen            answers (resumable); paced to 19,000 tokens per minute
  python results/live_best/mistral_large_test.py judge NAME     NAME = mistral|nemotron|glm45
  python results/live_best/mistral_large_test.py report         per dataset: good / wrong / handed off, latency, tokens, row-by-row agreement with the NVIDIA-hosted answers (offline best version, same rows)
Request: model mistral-large-4, temperature 0, top_p 1 (required with greedy sampling), reasoning_effort low, max_tokens 3000; the reply is a list of blocks, the answer is the text blocks.
"""
import concurrent.futures as cf
import importlib.util
import json
import os
import pathlib
import re
import statistics as st
import sys
import threading
import time
from collections import deque

import numpy as np
import requests

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent
cmd = sys.argv[1]
_argv = sys.argv[:]
sys.argv = ["x", "none"]
spec = importlib.util.spec_from_file_location("le", HERE / "live_eval.py")
le = importlib.util.module_from_spec(spec)
spec.loader.exec_module(le)
sys.argv = _argv
c = le.c
OUT = le.OUT
SCHEMA = ('Return only a JSON object with the keys: answer (string), confidence ("high", "medium" or "low"), needs_handoff (boolean), '
          'needs_human_review (boolean), cited_source (string). No text outside the JSON.')
GEN = OUT / "ml_gen.jsonl"
TPM = 19000  # pacing budget, tokens per minute
_win = deque()
_lock = threading.Lock()


def contexts():
    ctx = json.loads((RES / "embedding_test/work/e2e_contexts.json").read_text(encoding="utf-8"))["R2"]
    # the Wix tuning set and the first 100 TechQA questions; the held-out sets are not used here
    return {k: v for k, v in ctx.items() if v.get("set") in ("wix_first", "techqa")}


def reserve(tokens):
    """Block until `tokens` fit into the last 60 seconds' budget."""
    while True:
        with _lock:
            now = time.time()
            while _win and now - _win[0][0] > 60:
                _win.popleft()
            if sum(t for _, t in _win) + tokens <= TPM:
                _win.append((now, tokens))
                return
        time.sleep(1.5)


def ask(system, user):
    est = (len(system) + len(user)) // 4 + 700  # ~4 characters per token, plus ~700 tokens for reasoning and reply
    body = {"model": "mistral-large-4", "temperature": 0, "top_p": 1, "max_tokens": 3000, "reasoning_effort": "low", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    t0 = time.time()
    for attempt in range(10):
        reserve(est)
        try:
            r = requests.post("https://api.mistral.ai/v1/chat/completions", headers={"Authorization": "Bearer " + os.environ["MISTRAL_API_KEY"]}, json=body, timeout=180)
        except requests.RequestException:
            time.sleep(10)
            continue
        if r.status_code == 200:
            d = r.json()
            content = d["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
            return content or "", d.get("usage", {}).get("total_tokens", 0), round(time.time() - t0, 1)
        time.sleep(15 if r.status_code == 429 else 5)
    return "", 0, round(time.time() - t0, 1)


def parse(raw):
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except ValueError:
        d = {}
    return d


def gen():
    ctx = contexts()
    rows = le.jl(GEN)
    done = {r["key"] for r in rows if r.get("tokens")}
    GEN.write_text("".join(json.dumps(r) + "\n" for r in rows if r.get("tokens")), encoding="utf-8")
    todo = [k for k in ctx if k not in done]
    print("rows", len(ctx), "to answer", len(todo), flush=True)

    def one(k):
        x = ctx[k]
        raw, tok, secs = ask(x["system"] + "\n\n" + SCHEMA, f'Question:\n{x["question"]}\n\nSearch results (the documentation passages found for this question):\n{x["context"]}')
        d = parse(raw)
        sent = bool(d.get("answer")) and d.get("confidence") != "low" and d.get("needs_human_review") is not True
        return {"key": k, "answer": d.get("answer") or "", "sent": sent, "conf": d.get("confidence"), "tokens": tok, "secs": secs}

    with cf.ThreadPoolExecutor(3) as ex:
        for r in ex.map(one, todo):
            with open(GEN, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("gen done; without answer:", [r["key"] for r in le.jl(GEN) if not r.get("tokens")], flush=True)


def judge(name):
    b = c.a.backend(name)
    ctx = contexts()
    out = OUT / f"ml_judge_{name}.jsonl"
    gens = {r["key"]: r for r in le.jl(GEN) if r.get("tokens") and r["answer"].strip()}
    for _ in range(4):
        kept = [r for r in le.jl(out) if r["score"] is not None and r["key"] in gens]
        out.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
        done = {r["key"] for r in kept}
        todo = [k for k in gens if k not in done and gens[k]["sent"]]
        if not todo:
            break
        for k in todo:
            try:
                sc, err = c.score_answer_correctness(b, ctx[k]["question"], ctx[k]["reference"], gens[k]["answer"]), None
            except Exception as e:  # noqa: BLE001
                sc, err = None, type(e).__name__
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": k, "score": sc, "error": err}) + "\n")
            time.sleep({"mistral": 1.2, "nemotron": 0.4}.get(name, 0.8))
        time.sleep(20)
    print("judge", name, len([r for r in le.jl(out) if r["score"] is not None]), "of", len([k for k in gens if gens[k]["sent"]]), flush=True)


def report():
    ctx = contexts()
    g = {r["key"]: r for r in le.jl(GEN) if r.get("tokens")}
    s = {j: {r["key"]: r["score"] for r in le.jl(OUT / f"ml_judge_{j}.jsonl") if r["score"] is not None} for j in le.JUDGES}
    # the NVIDIA-hosted answers of the offline best version on the same rows
    nv = {r["key"]: r for r in le.jl(RES / "embedding_test/work/e2e_gen_R2.jsonl") if r.get("tokens")}
    nv.update({r["key"]: r for r in le.jl(RES / "answer_check/work/fresh_gen.jsonl") if r.get("tokens")})
    sn = {j: {r["key"]: r["score"] for r in le.jl(RES / f"embedding_test/work/e2e_judge_{j}_R2.jsonl") if r["score"] is not None} for j in le.JUDGES}
    for j in le.JUDGES:
        sn[j].update({r["key"]: r["score"] for r in le.jl(RES / f"answer_check/work/fresh_judge_{j}.jsonl") if r["score"] is not None})
    pg = lambda sc, k: sum((sc[j].get(k) or 0) >= 4 for j in le.JUDGES) >= 2  # noqa: E731
    print(f"{'':34s} | rows | good | wrong | handed off | wrong share of sent | score | corr (answered) | tokens/row | latency median / 90th pct")
    for label, ks in (("Wix 100", [k for k in ctx if k.startswith("wix")]), ("TechQA 100", [k for k in ctx if k.startswith("techqa")])):
        for name, gg, sc in (("Mistral-hosted mistral-large-4", g, s), ("NVIDIA-hosted (offline best)", nv, sn)):
            sent = [k for k in ks if k in gg and gg[k]["sent"]]
            gd = [k for k in sent if pg(sc, k)]
            wr = len(sent) - len(gd)
            corr = [st.mean(sc[j][k] for j in le.JUDGES if k in sc[j]) for k in sent if any(k in sc[j] for j in le.JUDGES)]
            secs = [gg[k]["secs"] for k in ks if k in gg and gg[k].get("secs")]
            toks = [gg[k]["tokens"] for k in ks if k in gg]
            print(f"{label + ' ' + name:34s} | {len(ks)} | {len(gd)} | {wr} | {len(ks) - len(sent)} | {100 * wr / max(1, len(sent)):.1f}% | {(len(gd) - wr) / len(ks):+.3f} | {np.mean(corr):.2f} | {np.mean(toks):.0f} | {np.median(secs):.0f}s / {np.percentile(secs, 90):.0f}s")
        both = [k for k in ks if k in g and k in nv]
        same = sum((g[k]["sent"] == nv[k]["sent"]) and (not g[k]["sent"] or pg(s, k) == pg(sn, k)) for k in both)
        print(f"  {label}: same outcome (handed off, or sent and same good/wrong verdict) on {same} of {len(both)} rows")
        miss = [k for k in ks if g.get(k, {}).get("sent") and sum(k in s[j] for j in le.JUDGES) < 3]
        print(f"  {label}: answered rows missing a grade: {len(miss)}")


if __name__ == "__main__":
    {"gen": gen, "judge": lambda: judge(sys.argv[2]), "report": report}[cmd]()
