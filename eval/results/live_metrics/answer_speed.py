"""How long does the answer call take on an endpoint that is not congested?
The live workflow's prompt (system prompt, question and passages, reasoning effort low, temperature 0) goes to Mistral's own API for mistral-large-4, one request at a time, so nothing waits in a queue.
Only the HTTP request is timed, not the pauses between requests. 15 Wix and 15 TechQA questions, random with a fixed seed.
usage (from eval/, MISTRAL_API_KEY loaded): python results/live_metrics/answer_speed.py run | report"""
import json
import os
import pathlib
import random
import statistics
import sys
import time

import requests

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "answer_speed.jsonl"
CTX = json.loads((HERE.parent / "embedding_test/work/e2e_contexts.json").read_text(encoding="utf-8"))["R2"]
SCHEMA = ('Return only a JSON object with the keys: answer (string), confidence ("high", "medium" or "low"), needs_handoff (boolean), '
          'needs_human_review (boolean), cited_source (string). No text outside the JSON.')


def user_message(v):
    return "Question:\n" + v["question"] + "\n\nSearch results (the documentation passages found for this question):\n" + v["context"]


def pick():
    rnd = random.Random(11)
    out = []
    for ds in ("wix", "techqa"):
        ks = sorted(k for k in CTX if k.startswith(ds))
        out += rnd.sample(ks, 15)
    return out


def run():
    done = {json.loads(x)["key"] for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip()} if OUT.exists() else set()
    for k in pick():
        if k in done:
            continue
        v = CTX[k]
        system = v["system"] + "\n\n" + SCHEMA if SCHEMA not in v["system"] else v["system"]
        body = {"model": "mistral-large-4", "temperature": 0, "top_p": 1, "max_tokens": 3000, "reasoning_effort": "low",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_message(v)}]}
        for attempt in range(5):
            t0 = time.time()
            r = requests.post("https://api.mistral.ai/v1/chat/completions", headers={"Authorization": "Bearer " + os.environ["MISTRAL_API_KEY"]}, json=body, timeout=240)
            secs = time.time() - t0
            if r.status_code == 200:
                u = r.json().get("usage", {})
                rec = {"key": k, "secs": round(secs, 2), "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens")}
                with open(OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(rec, flush=True)
                break
            print("retry", r.status_code, flush=True)
            time.sleep(20)
        time.sleep(14)  # pause between requests; not part of the timing


def report():
    rows = [json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip()]
    for ds in ("wix", "techqa"):
        r = [x for x in rows if x["key"].startswith(ds)]
        s = sorted(x["secs"] for x in r)
        rate = [x["completion_tokens"] / x["secs"] for x in r if x.get("completion_tokens")]
        print(f"{ds}: {len(r)} calls | answer call median {statistics.median(s):.1f} s, mean {statistics.fmean(s):.1f} s, 90th pct {s[int(.9 * len(s))]:.1f} s, max {s[-1]:.1f} s | "
              f"input tokens mean {statistics.fmean(x['prompt_tokens'] for x in r):,.0f}, output tokens mean {statistics.fmean(x['completion_tokens'] for x in r):,.0f} | output speed about {statistics.median(rate):.0f} tokens/s")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
