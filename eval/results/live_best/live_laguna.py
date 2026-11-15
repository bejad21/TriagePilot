"""Grade the answered rows of the live run (200 Wix + 200 TechQA) with Laguna, the third judge of the panel (nemotron, glm45, laguna).
usage (from eval/, keys loaded): python results/live_best/live_laguna.py run | report     (run writes work/judge_laguna.jsonl in the same format as the other judge files; report prints the panel result and how the judges agree)"""
import concurrent.futures as cf
import importlib.util
import json
import pathlib
import re
import sys
import time

import requests

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness import nvidia_key, prompts  # noqa: E402

cmd = sys.argv[1]
sys.argv = [sys.argv[0], "none"]
spec = importlib.util.spec_from_file_location("live_eval", HERE / "live_eval.py")
le = importlib.util.module_from_spec(spec)
spec.loader.exec_module(le)
OUT = HERE / "work/judge_laguna.jsonl"


def answered():
    R = le.rows()
    res = {}
    for ds in ("wix", "techqa"):
        for r in le.jl(HERE / f"work/live_{ds}.jsonl"):
            if r["status"] == "answered" and r["answer"].strip():
                res[r["key"]] = (R[r["key"]]["reference"], r["answer"])
    return res


def ask(k, ref, ans):
    body = {"model": "poolside/laguna-xs-2.1", "temperature": 0, "max_tokens": 2500, "messages": [{"role": "user", "content": prompts.ANSWER_CORRECTNESS.format(reference=ref, answer=ans)}]}
    score, err = None, None
    for _ in range(6):
        try:
            r = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers={"Authorization": "Bearer " + nvidia_key.api_key()}, json=body, timeout=120)
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
    return {"key": k, "score": score, "error": err}


def run():
    a = answered()
    kept = [r for r in le.jl(OUT) if r["score"] is not None and r["key"] in a]
    OUT.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
    todo = [k for k in a if k not in {r["key"] for r in kept}]
    print("answered rows:", len(a), "| to grade:", len(todo), flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(6) as ex:
        for r in ex.map(lambda k: ask(k, *a[k]), todo):
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
    print(f"done in {time.time() - t0:.0f} s", flush=True)


def report():
    R = le.rows()
    a = answered()
    judges = ("nemotron", "glm45", "laguna")
    S = {j: {r["key"]: r["score"] for r in le.jl(HERE / f"work/judge_{j}.jsonl") if r["score"] is not None} for j in judges}
    print(f"graded by laguna: {len(S['laguna'])} of {len(a)} answered rows\n")
    print("dataset | rows | answered | handed off | good sent | wrong sent | wrong share of sent | score (good - wrong)/rows")
    for ds in ("wix", "techqa"):
        keys = [k for k in R if R[k]["ds"] == ds]
        ans = [k for k in keys if k in a]
        good = {k: sum((S[j].get(k) or 0) >= 4 for j in judges) >= 2 for k in ans}
        gs = sum(good.values())
        ws = len(ans) - gs
        print(f"{ds} | {len(keys)} | {len(ans)} | {len(keys) - len(ans)} | {gs} | {ws} | {100 * ws / max(1, len(ans)):.1f}% | {(gs - ws) / len(keys):+.3f}")
    print("\nhow often each judge says good (score >= 4) on the answered rows:")
    for j in S:
        ks = [k for k in a if k in S[j]]
        print(f"  {j:9s} {sum(S[j][k] >= 4 for k in ks)} of {len(ks)} | mean score {sum(S[j][k] for k in ks) / len(ks):.2f}")
    print("\nlaguna vs each other judge (same good/not-good verdict):")
    for j in ("nemotron", "glm45"):
        ks = [k for k in a if k in S[j] and k in S["laguna"]]
        print(f"  vs {j:9s} {sum((S[j][k] >= 4) == (S['laguna'][k] >= 4) for k in ks)} of {len(ks)} | exact same 1-5 score {sum(S[j][k] == S['laguna'][k] for k in ks)}")


if __name__ == "__main__":
    {"run": run, "report": report}[cmd]()
