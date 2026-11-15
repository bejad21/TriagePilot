"""Does a higher reasoning effort of mistral-large-4 help? Rows: every wrong answer and every hand-off of the live run, plus 26 random good answers (13 per dataset, seed 7). The rows lean towards hard cases, so scores are lower than on the full set.
The live workflow's prompt goes to Mistral's own API at reasoning effort low, medium and high (temperature 0); only the effort changes. Passages are rebuilt as in live_metrics.
Graded by the current panel (nemotron, glm45, laguna; good = 2 of 3 give 4 or more). An answer is sent unless confidence is low or needs_human_review is true, as in the workflow's escalation rule.
Wix and TechQA are reported separately.
usage (from eval/, Mistral, NVIDIA and Z.ai keys loaded): python results/effort_test/effort_test.py select | run EFFORT | grade JUDGE | report     EFFORT = low | medium | high, JUDGE = nemotron | glm45 | laguna
EFFORT_JUDGES (comma-separated judge names) restricts the judges `report` uses; with fewer than three, good = every judge gives 4 or more."""
import concurrent.futures as cf
import importlib.util
import json
import os
import pathlib
import queue
import random
import re
import statistics
import sys
import threading
import time
from collections import Counter

import requests

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent
sys.path.insert(0, str(HERE.parents[1]))
from harness import judges, panel  # noqa: E402

ARGS = sys.argv[:]
cmd = sys.argv[1]
sys.argv = ["x", "none"]
spec = importlib.util.spec_from_file_location("live_eval", RES / "live_best/live_eval.py")
le = importlib.util.module_from_spec(spec)
spec.loader.exec_module(le)
LIVEW = RES / "live_best/work"
CTXF = RES / "live_metrics/contexts.json"
ROWS = HERE / "rows.json"
SCHEMA = ('Return only a JSON object with the keys: answer (string), confidence ("high", "medium" or "low"), needs_handoff (boolean), '
          'needs_human_review (boolean), cited_source (string). No text outside the JSON.')
KEY = os.environ.get("MISTRAL_API_KEY")


def jl(p):
    return le.jl(p)


def live_outcomes():
    R = le.rows()
    S = {j: {r["key"]: r["score"] for r in jl(LIVEW / f"judge_{j}.jsonl") if r["score"] is not None} for j in panel.PANEL}
    out = {}
    for ds in ("wix", "techqa"):
        for r in jl(LIVEW / f"live_{ds}.jsonl"):
            k = r["key"]
            if r["status"] == "answered" and r["answer"].strip():
                out[k] = "good" if panel.is_good([S[j].get(k) for j in panel.PANEL]) else "wrong"
            else:
                out[k] = "handed off"
    return R, out


def select():
    R, out = live_outcomes()
    rnd = random.Random(7)
    rows = {}
    for ds in ("wix", "techqa"):
        ks = [k for k in R if R[k]["ds"] == ds]
        for k in ks:
            if out[k] in ("wrong", "handed off"):
                rows[k] = out[k]
        good = sorted(k for k in ks if out[k] == "good")
        for k in rnd.sample(good, 13):
            rows[k] = "good"
    ROWS.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(len(rows), "rows:", {ds: dict(Counter(v for k, v in rows.items() if k.startswith(ds))) for ds in ("wix", "techqa")})


def systems():
    base = json.loads((RES / "embedding_test/work/e2e_contexts.json").read_text(encoding="utf-8"))["R2"]
    out = {}
    for ds in ("wix", "techqa"):
        k = next(k for k in base if k.startswith(ds))
        s = base[k]["system"]
        out[ds] = s if SCHEMA in s else s + "\n\n" + SCHEMA
    return out


def parse(raw):
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else {}
    except (ValueError, AttributeError):
        return {}


def run(effort):
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    R = le.rows()
    ctx = json.loads(CTXF.read_text(encoding="utf-8"))
    sysm = systems()
    out = HERE / f"gen_{effort}.jsonl"
    done = {r["key"] for r in jl(out) if r.get("ok")}
    todo = [k for k in rows if k not in done]
    print(effort, "to answer:", len(todo), flush=True)
    q = queue.Queue()
    for k in todo:
        q.put(k)
    lock = threading.Lock()

    def worker(key):
        last = 0.0
        while True:
            try:
                k = q.get_nowait()
            except queue.Empty:
                return
            ds = R[k]["ds"]
            body = {"model": "mistral-large-4", "temperature": 0, "top_p": 1, "max_tokens": 8000, "reasoning_effort": effort,
                    "messages": [{"role": "system", "content": sysm[ds]},
                                 {"role": "user", "content": "Question:\n" + R[k]["question"] + "\n\nSearch results (the documentation passages found for this question):\n" + ctx[k]["context"]}]}
            rec = {"key": k, "ok": False}
            for attempt in range(8):
                wait = last + 16 - time.time()  # pause between calls; not part of the timing
                if wait > 0:
                    time.sleep(wait)
                t0 = time.time()
                last = t0
                try:
                    r = requests.post("https://api.mistral.ai/v1/chat/completions", headers={"Authorization": "Bearer " + key}, json=body, timeout=420)
                except requests.RequestException:
                    time.sleep(10)
                    continue
                secs = time.time() - t0
                if r.status_code == 200:
                    d = r.json()
                    c = d["choices"][0]["message"]["content"]
                    if isinstance(c, list):
                        c = "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
                    u = d.get("usage", {})
                    p = parse(c)
                    sent = bool(str(p.get("answer", "")).strip()) and p.get("confidence") != "low" and p.get("needs_human_review") is not True
                    rec = {"key": k, "ok": True, "secs": round(secs, 2), "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                           "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"), "sent": sent, "answer": str(p.get("answer", "")), "confidence": p.get("confidence")}
                    break
                time.sleep(20 if r.status_code == 429 else 6)
            with lock:
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

    t0 = time.time()
    ts = [threading.Thread(target=worker, args=(KEY,))] if KEY else []
    [t.start() for t in ts]
    [t.join() for t in ts]
    got = [r for r in jl(out) if r.get("ok")]
    print(effort, "done:", len({r['key'] for r in got}), "of", len(rows), "in", round(time.time() - t0), "s", flush=True)


def grade(judge):
    R = le.rows()
    for effort in ("low", "medium", "high"):
        gens = {r["key"]: r for r in jl(HERE / f"gen_{effort}.jsonl") if r.get("ok") and r["sent"]}
        for j in (judge,):
            out = HERE / f"judge_{j}_{effort}.jsonl"
            have = {r["key"] for r in jl(out) if r["score"] is not None}
            todo = [k for k in gens if k not in have]
            b = panel.build(j)

            def one(k):
                try:
                    return {"key": k, "score": judges.score_answer_correctness(b, R[k]["question"], R[k]["reference"], gens[k]["answer"]), "error": None}
                except Exception as e:  # noqa: BLE001
                    return {"key": k, "score": None, "error": f"{type(e).__name__}"}

            with cf.ThreadPoolExecutor(3) as ex:
                for r in ex.map(one, todo):
                    with open(out, "a", encoding="utf-8") as f:
                        f.write(json.dumps(r) + "\n")
            n = len({r['key'] for r in jl(out) if r['score'] is not None})
            print("graded", effort, j, n, "of", len(gens), flush=True)


def report():
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    lines = []
    arms = {}
    for effort in ("low", "medium", "high"):
        gens = {r["key"]: r for r in jl(HERE / f"gen_{effort}.jsonl") if r.get("ok")}
        use = tuple(j for j in panel.PANEL if j in os.environ.get("EFFORT_JUDGES", ",".join(panel.PANEL)).split(","))
        S = {j: {r["key"]: r["score"] for r in jl(HERE / f"judge_{j}_{effort}.jsonl") if r["score"] is not None} for j in use}
        oc = {}
        for k in rows:
            g = gens.get(k)
            if not g:
                oc[k] = "missing"
            elif not g["sent"]:
                oc[k] = "handed off"
            else:
                sc = [S[j].get(k) for j in use]
                oc[k] = "ungraded" if sum(s is not None for s in sc) < min(2, len(use)) else ("good" if (panel.is_good(sc) if len(use) == 3 else all(s is not None and s >= 4 for s in sc)) else "wrong")
        arms[effort] = (gens, oc)
    live = live_outcomes()[1]
    for ds, label in (("wix", "Wix"), ("techqa", "TechQA")):
        ks = [k for k in rows if k.startswith(ds)]
        lines += [f"## {label}: {len(ks)} questions ({Counter(rows[k] for k in ks)['wrong']} wrong and {Counter(rows[k] for k in ks)['handed off']} handed off in the live run, {Counter(rows[k] for k in ks)['good']} random good)", "",
                  "| Setup | Good sent | Wrong sent | Handed off | Score on these rows | Not graded |", "|---|---|---|---|---|---|"]
        c = Counter(live[k] for k in ks)
        lines.append(f"| Live run (shared free endpoint, low) | {c['good']} | {c['wrong']} | {c['handed off']} | {(c['good'] - c['wrong']) / len(ks):+.2f} | 0 |")
        for effort in ("low", "medium", "high"):
            oc = arms[effort][1]
            c = Counter(oc[k] for k in ks)
            lines.append(f"| Mistral API, {effort} | {c['good']} | {c['wrong']} | {c['handed off']} | {(c['good'] - c['wrong']) / len(ks):+.2f} | {c['missing'] + c['ungraded']} |")
        lines += ["", "| Compared with low | Wrong turned good | Good turned wrong | Wrong turned hand-off | Hand-off turned good | Hand-off turned wrong |", "|---|---|---|---|---|---|"]
        for effort in ("medium", "high"):
            lo, hi = arms["low"][1], arms[effort][1]
            mv = Counter((lo[k], hi[k]) for k in ks)
            lines.append(f"| {effort} | {mv[('wrong', 'good')]} | {mv[('good', 'wrong')]} | {mv[('wrong', 'handed off')]} | {mv[('handed off', 'good')]} | {mv[('handed off', 'wrong')]} |")
        lines.append("")
    lines += ["## Time and tokens per request (all 100 rows; Wix and TechQA combined only here because the call itself does not differ by dataset)", "",
              "| Effort | Time, median | Time, mean | Time, 90th pct | Input tokens | Output tokens | of which reasoning | Total vs low |", "|---|---|---|---|---|---|---|---|"]
    base = None
    for effort in ("low", "medium", "high"):
        gens = [r for r in arms[effort][0].values()]
        s = sorted(r["secs"] for r in gens)
        pt = statistics.fmean(r["prompt_tokens"] for r in gens)
        ct = statistics.fmean(r["completion_tokens"] for r in gens)
        rt = statistics.fmean(r["reasoning_tokens"] or 0 for r in gens)
        tot = pt + ct
        base = base or tot
        lines.append(f"| {effort} | {statistics.median(s):.1f} s | {statistics.fmean(s):.1f} s | {s[int(.9 * len(s))]:.1f} s | {pt:,.0f} | {ct:,.0f} | {rt:,.0f} | {tot / base:.2f}x |")
    lines += ["", "Cost per 1,000 requests at the two quoted list prices ($0.70 / $2.10 and $1.36 / $4.18 per million input / output tokens):", "", "| Effort | Lower price | Higher price |", "|---|---|---|"]
    for effort in ("low", "medium", "high"):
        gens = list(arms[effort][0].values())
        pt = statistics.fmean(r["prompt_tokens"] for r in gens)
        ct = statistics.fmean(r["completion_tokens"] for r in gens)
        lines.append(f"| {effort} | ${(pt * 0.70 + ct * 2.10) / 1000:.2f} | ${(pt * 1.36 + ct * 4.18) / 1000:.2f} |")
    note = f"Judges used for every row in this report: {', '.join(use)}. With two judges an answer counts as good only if both give it 4 or more (stricter than the usual 2 of 3). The third judge (laguna) did not respond during this test, so it is left out of all three efforts.\n\n" if len(use) < 3 else ""
    text = "# Reasoning effort test\n\nBuilt by `effort_test.py report`.\n\n" + note + "\n".join(lines) + "\n"
    (HERE / "README.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    {"select": select, "run": lambda: run(ARGS[2]), "grade": lambda: grade(ARGS[2]), "report": report}[cmd]()
