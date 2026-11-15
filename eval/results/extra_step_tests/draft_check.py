"""Would a second model that checks the draft before it is sent improve the score?
The checker is Laguna, a different model family than the answer model. It sees the question, the passages and the draft (not the reference) and rates 1-5 how safe the draft is to send.
For each strictness level, blocking a wrong answer gains a point (a person handles it) and blocking a good one loses a point.
Ground truth is the earlier panel (ministral-14b, nemotron, glm45), so Laguna does not grade itself.
Inputs: offline replays with saved passages (the 300 base answers R2 and the hybrid / reranker / noise-control arms). All wrong answers and 250 random good ones are checked; good ones are weighted back up.
usage (from eval/, NVIDIA key loaded): python results/extra_step_tests/draft_check.py run | report"""
import concurrent.futures as cf
import importlib.util
import json
import pathlib
import random
import re
import sys
import time

import requests

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent / "rerank_hybrid"
sys.path.insert(0, str(HERE.parents[1]))
from harness import nvidia_key, panel  # noqa: E402

spec = importlib.util.spec_from_file_location("lf", R / "laguna_full.py")
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "report"]
lf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lf)
OUT = HERE / "checker.jsonl"
PROMPT = ("You review a customer-support draft before it is sent. You see the customer's question, the documentation passages the assistant was given, and the assistant's draft answer.\n\n"
          "Customer question:\n{question}\n\nDocumentation passages:\n{context}\n\nDraft answer:\n{answer}\n\n"
          "Rate from 1 to 5 how safe it is to send this draft. 5 = every claim is supported by the passages and it answers what the customer asked. 3 = partly supported, or it answers only part of the question. "
          '1 = it contains claims that are not in the passages, contradicts them, or answers a different question. Return ONLY JSON: {{"score": <1-5 integer>}}.')


def contexts():
    base = json.loads((lf.E / "e2e_contexts.json").read_text(encoding="utf-8"))["R2"]
    arms = json.loads((lf.W / "arm_contexts.json").read_text(encoding="utf-8"))
    out = {}
    for k, v in base.items():
        out[f"R2|{k}"] = (v["question"], v["context"])
    for a, d in arms.items():
        for k, v in d.items():
            out[f"{a}|{k}"] = (v["question"], v["context"])
    return out


def ground_truth():
    units = {u: d for u, d in lf.collect().items() if d["sent"] and all(d["scores"][j] is not None for j in panel.OLD_PANEL)}
    ctx = contexts()
    return {u: (d, ctx[u]) for u, d in units.items() if u in ctx}


def ask(u, d, qc):
    q, c = qc
    body = {"model": "poolside/laguna-xs-2.1", "temperature": 0, "max_tokens": 2500,
            "messages": [{"role": "user", "content": PROMPT.format(question=q[:3000], context=c[:40000], answer=d["answer"])}]}
    score, err = None, None
    for _ in range(6):
        try:
            r = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers={"Authorization": "Bearer " + nvidia_key.api_key()}, json=body, timeout=240)
        except Exception as e:  # noqa: BLE001
            err = type(e).__name__
            time.sleep(5)
            continue
        if r.status_code == 200:
            m = re.search(r"\{.*\}", r.json()["choices"][0]["message"].get("content") or "", re.S)
            try:
                v = json.loads(m.group(0))["score"]
                score, err = (int(v) if v == int(v) and 1 <= v <= 5 else None), None
            except (ValueError, KeyError, TypeError, AttributeError):
                err = "parse"
            break
        err = f"http{r.status_code}"
        time.sleep(15 if r.status_code == 429 else 5)
    return {"unit": u, "score": score, "err": err}


def sample(gt):
    good = {u: panel.is_good([d["scores"][j] for j in panel.OLD_PANEL]) for u, (d, _) in gt.items()}
    wrong = [u for u in gt if not good[u]]
    goods = [u for u in gt if good[u]]
    random.Random(5).shuffle(goods)
    return good, wrong, goods[:250], len(goods)


def run():
    gt = ground_truth()
    good, wrong, gsample, ngood = sample(gt)
    todo_all = wrong + gsample
    done = {r["unit"] for r in lf.jl(OUT) if r["score"] is not None}
    OUT.write_text("".join(json.dumps(r) + "\n" for r in lf.jl(OUT) if r["score"] is not None), encoding="utf-8")
    todo = [u for u in todo_all if u not in done]
    print(f"answers with passages and all panel grades: {len(gt)} | wrong {len(wrong)} | good {ngood} (checking {len(gsample)}) | to check now: {len(todo)}", flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(6) as ex:
        for r in ex.map(lambda u: ask(u, *gt[u]), todo):
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
    print(f"done in {time.time() - t0:.0f} s", flush=True)


def report():
    gt = ground_truth()
    good, wrong, gsample, ngood = sample(gt)
    sc = {r["unit"]: r["score"] for r in lf.jl(OUT) if r["score"] is not None}
    w = [u for u in wrong if u in sc]
    g = [u for u in gsample if u in sc]
    wt = ngood / max(1, len(g))
    print(f"answers sent: {len(gt)} = {len(wrong)} wrong + {ngood} good (by the earlier panel). Checked: {len(w)} wrong, {len(g)} good (each good one stands for {wt:.2f}).")
    print("score the checker gives: wrong answers", sorted((s, sum(sc[u] == s for u in w)) for s in range(1, 6)), "| good answers (sample)", sorted((s, sum(sc[u] == s for u in g)) for s in range(1, 6)))
    print("\nblock an answer if the checker's score is below X:")
    print("X | wrong answers blocked (+1 each) | good answers blocked (-1 each, scaled to all good) | net points | net points per 100 answers sent")
    for x in (2, 3, 4, 5):
        wb = sum(sc[u] < x for u in w)
        gb = sum(sc[u] < x for u in g) * wt
        net = wb - gb
        print(f"{x} | {wb} of {len(w)} ({100 * wb / len(w):.0f}%) | {gb:.0f} of {ngood} ({100 * gb / ngood:.1f}%) | {net:+.0f} | {net / len(gt) * 100:+.1f}")
    pairs = [(sc[u], 0) for u in w] + [(sc[u], 1) for u in g]
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    auc = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    print(f"\nhow well the checker separates good from wrong (AUC, 0.5 = coin flip): {auc:.2f}")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
