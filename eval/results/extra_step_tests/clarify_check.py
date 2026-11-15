"""Would "ask the customer a clarifying question when the question is too vague" help? Laguna rates each of the 400 live questions (question text only) from 1 (too vague, an agent would have to ask what the customer means) to 5 (clear).
The vague questions are then tallied by live outcome (good answer, wrong answer, handed off).
Asking back scores 0 for a question: +1 when it replaces a wrong answer, -1 when it replaces a good one, no change when the case was handed off anyway.
usage (from eval/, NVIDIA key loaded): python results/extra_step_tests/clarify_check.py run | report"""
import concurrent.futures as cf
import importlib.util
import json
import pathlib
import re
import sys
import time

import requests

HERE = pathlib.Path(__file__).resolve().parent
LIVE = HERE.parent / "live_best"
sys.path.insert(0, str(HERE.parents[1]))
from harness import nvidia_key, panel  # noqa: E402

cmd = sys.argv[1]
sys.argv = ["x", "none"]
spec = importlib.util.spec_from_file_location("ll", LIVE / "live_laguna.py")
ll = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ll)
le = ll.le
OUT = HERE / "clarify.jsonl"
PROMPT = ("A customer wrote this question to a software support desk. A support agent can search the product documentation, but cannot talk to the customer first.\n\nCustomer question:\n{q}\n\n"
          "Rate from 1 to 5 whether the agent could give a reliable answer from the documentation as the question is written. 5 = clear and specific. 3 = somewhat vague, but a reasonable answer is possible. "
          '1 = too vague or ambiguous: the agent would first have to ask what the customer means (for example which product, version, page type or feature). Return ONLY JSON: {{"score": <1-5 integer>}}.')


def ask(k, q):
    body = {"model": "poolside/laguna-xs-2.1", "temperature": 0, "max_tokens": 2500, "messages": [{"role": "user", "content": PROMPT.format(q=q[:3000])}]}
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
    return {"key": k, "score": score, "err": err}


def run():
    R = le.rows()
    keys = [k for ds in ("wix", "techqa") for k in R if R[k]["ds"] == ds and k in {r["key"] for r in le.jl(LIVE / f"work/live_{ds}.jsonl")}]
    done = {r["key"] for r in le.jl(OUT) if r["score"] is not None}
    OUT.write_text("".join(json.dumps(r) + "\n" for r in le.jl(OUT) if r["score"] is not None), encoding="utf-8")
    todo = [k for k in keys if k not in done]
    print("questions:", len(keys), "| to rate:", len(todo), flush=True)
    with cf.ThreadPoolExecutor(6) as ex:
        for r in ex.map(lambda k: ask(k, R[k]["question"]), todo):
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
    print("done", flush=True)


def report():
    a = ll.answered()
    S = {j: {r["key"]: r["score"] for r in le.jl(LIVE / f"work/judge_{j}.jsonl") if r["score"] is not None} for j in ("nemotron", "glm45", "laguna")}
    sc = {r["key"]: r["score"] for r in le.jl(OUT) if r["score"] is not None}
    outcome = {}
    for k in sc:
        outcome[k] = "handed off" if k not in a else ("good" if panel.is_good([S[j].get(k) for j in panel.PANEL]) else "wrong")
    print("questions rated:", len(sc), "| outcomes:", {o: sum(v == o for v in outcome.values()) for o in ("good", "wrong", "handed off")})
    print("\nclarity score of the question (1 = too vague, 5 = clear) by what the live system did:")
    for o in ("good", "wrong", "handed off"):
        ks = [k for k in sc if outcome[k] == o]
        print(f"  {o:10s} n={len(ks):3d} | " + ", ".join(f"{s}: {sum(sc[k] == s for k in ks)}" for s in range(1, 6)))
    print("\nask the customer instead of answering when the clarity score is below X:")
    print("X | wrong answers replaced (+1 each) | good answers replaced (-1 each) | handed-off cases touched (0) | net points on 400 questions")
    for x in (2, 3, 4):
        w = sum(1 for k in sc if outcome[k] == "wrong" and sc[k] < x)
        g = sum(1 for k in sc if outcome[k] == "good" and sc[k] < x)
        h = sum(1 for k in sc if outcome[k] == "handed off" and sc[k] < x)
        print(f"{x} | {w} of {sum(v == 'wrong' for v in outcome.values())} | {g} of {sum(v == 'good' for v in outcome.values())} | {h} | {w - g:+d}  (score change {(w - g) / len(sc):+.3f})")
    print("\nthe four rows expected to be vague:", {k: sc.get(k) for k in ("wix:127", "wix:168", "techqa:36", "techqa:52")})
    print("lowest-rated good answers (these would be lost):", [(k, sc[k]) for k in sorted((k for k in sc if outcome[k] == "good"), key=lambda k: sc[k])[:8]])


if __name__ == "__main__":
    {"run": run, "report": report}[cmd]()
