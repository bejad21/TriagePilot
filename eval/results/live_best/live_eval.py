"""Send 200 Wix and 200 TechQA questions to the LIVE webhooks (triage-wix, triage-techqa), grade the answered ones with the three judges, report per dataset.
usage (from eval/, keys loaded incl. TRIAGEPILOT_WEBHOOK_KEY and judge keys):
  python results/live_best/live_eval.py run DS        DS = wix | techqa     (resumable; WORKERS env, default 3)
  python results/live_best/live_eval.py judge NAME    NAME = mistral|nemotron|glm45 (grades answered rows of both datasets; resumable)
  python results/live_best/live_eval.py report
Rows: Wix = ids_wix.txt (100) + ids_wix_fresh.txt (100); TechQA = ids 31 to 130 (the original 100) + 131 to 230 (the fresh 100). Question text and references are the saved ones of the earlier offline tests.
Status from the webhook: answered (graded), escalated (handed off), rejected / error / timeout (counted separately; errors are retried twice).
"""
import concurrent.futures as cf
import importlib.util
import json
import os
import pathlib
import random
import statistics as st
import sys
import time

import numpy as np
import requests

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent
EVAL = RES.parent
EMB = RES / "embedding_test/work"
CHK = RES / "answer_check/work"
OUT = HERE / "work"
OUT.mkdir(exist_ok=True)
URL = {"wix": "http://localhost:5678/webhook/triage-wix", "techqa": "http://localhost:5678/webhook/triage-techqa"}
JUDGES = ("mistral", "nemotron", "glm45")
cmd = sys.argv[1]
_argv = sys.argv[:]
sys.argv = [sys.argv[0], "none"]
spec = importlib.util.spec_from_file_location("checks40", RES / "answer_check/checks40.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
sys.argv = _argv


def jl(p):
    p = pathlib.Path(p)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def rows():
    """Return {key: {ds, question, reference}}."""
    ctx = json.loads((EMB / "e2e_contexts.json").read_text(encoding="utf-8"))["R2"]
    fresh = json.loads((CHK / "fresh_ctx.json").read_text(encoding="utf-8"))
    out = {}
    for k, v in list(ctx.items()) + list(fresh.items()):
        out[k] = {"ds": k.split(":")[0], "question": v["question"], "reference": v["reference"]}
    return out


def run(ds):
    R = {k: v for k, v in rows().items() if v["ds"] == ds}
    p = OUT / f"live_{ds}.jsonl"
    done = {r["key"] for r in jl(p) if r["status"] in ("answered", "escalated", "rejected")}
    p.write_text("".join(json.dumps(r) + "\n" for r in jl(p) if r["status"] in ("answered", "escalated", "rejected")), encoding="utf-8")
    key = os.environ["TRIAGEPILOT_WEBHOOK_KEY"]

    def one(k):
        t0 = time.time()
        res = {"key": k, "status": "error", "answer": "", "cited_source": "", "secs": 0}
        for attempt in range(3):
            try:
                r = requests.post(URL[ds], json={"text": R[k]["question"]}, headers={"X-TriagePilot-Key": key}, timeout=600)
                if r.status_code == 200 and r.text.strip():
                    d = r.json()
                    res.update(status=d.get("status", "error"), answer=d.get("answer", ""), cited_source=d.get("cited_source", ""))
                    if res["status"] in ("answered", "escalated", "rejected"):
                        break
                else:
                    res["status"] = f"http_{r.status_code}"
            except (requests.RequestException, ValueError) as e:
                res["status"] = type(e).__name__
            time.sleep(10 * (attempt + 1))
        res["secs"] = round(time.time() - t0, 1)
        time.sleep(float(os.environ.get("PAUSE", "0")))
        return res

    todo = [k for k in R if k not in done]
    print(ds, "rows", len(R), "to send", len(todo), flush=True)
    with cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", "3"))) as ex:
        for res in ex.map(one, todo):
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
    got = {r["key"]: r for r in jl(p)}
    from collections import Counter
    print(ds, "done:", dict(Counter(r["status"] for r in got.values())), "missing", len(R) - len(got), flush=True)


def judge(name):
    b = c.a.backend(name)
    R = rows()
    out = OUT / f"judge_{name}.jsonl"
    ans = {}
    for ds in ("wix", "techqa"):
        for r in jl(OUT / f"live_{ds}.jsonl"):
            if r["status"] == "answered" and r["answer"].strip():
                ans[r["key"]] = r["answer"]
    for _ in range(4):
        kept = [r for r in jl(out) if r["score"] is not None and r["key"] in ans]
        out.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
        todo = [k for k in ans if k not in {r["key"] for r in kept}]
        if not todo:
            break
        for k in todo:
            try:
                sc, err = c.score_answer_correctness(b, R[k]["question"], R[k]["reference"], ans[k]), None
            except Exception as e:  # noqa: BLE001
                sc, err = None, type(e).__name__
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": k, "score": sc, "error": err}) + "\n")
            time.sleep({"mistral": 1.2, "nemotron": 0.4}.get(name, 0.8))
        time.sleep(20)
    print("judge", name, len([r for r in jl(out) if r["score"] is not None]), "of", len(ans), flush=True)


def ci(vals, n=5000):
    rnd = random.Random(0)
    v = list(vals)
    bs = [float(np.mean([v[rnd.randrange(len(v))] for _ in v])) for _ in range(n)]
    return float(np.mean(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def report():
    R = rows()
    s = {j: {r["key"]: r["score"] for r in jl(OUT / f"judge_{j}.jsonl") if r["score"] is not None} for j in JUDGES}
    pp = lambda x: f"{100 * x[0]:.1f}% ({100 * x[1]:.1f} to {100 * x[2]:.1f})"  # noqa: E731
    table = {}
    for ds, label in (("wix", "Wix (200)"), ("techqa", "TechQA (200)")):
        got = {r["key"]: r for r in jl(OUT / f"live_{ds}.jsonl")}
        keys = [k for k in R if R[k]["ds"] == ds]
        ans = [k for k in keys if got.get(k, {}).get("status") == "answered" and got[k]["answer"].strip()]
        miss = [k for k in ans if sum(k in s[j] for j in JUDGES) < 3]
        good = {k: c.pg([s[j].get(k) for j in JUDGES]) for k in ans}
        corr = {k: st.mean(s[j][k] for j in JUDGES if k in s[j]) for k in ans}
        status = {k: got.get(k, {}).get("status", "missing") for k in keys}
        n = len(keys)
        gs = sum(good[k] for k in ans)
        ws = len(ans) - gs
        sc = ci([(1 if good[k] else -1) if k in good else 0 for k in keys])
        secs = [got[k]["secs"] for k in keys if k in got]
        table[label] = [("Rows with a response", f"{sum(1 for k in keys if k in got)} of {n}"),
                        ("Answered / escalated / rejected / other", f"{len(ans)} / {sum(v == 'escalated' for v in status.values())} / {sum(v == 'rejected' for v in status.values())} / {n - len(ans) - sum(v in ('escalated', 'rejected') for v in status.values())}"),
                        ("Good answers sent", f"{gs} ({100 * gs / n:.1f}% of rows)"), ("Wrong answers sent", f"{ws} ({100 * ws / max(1, len(ans)):.1f}% of sent)"),
                        ("Wrong share of sent, 95% interval", pp(ci([0 if good[k] else 1 for k in ans]))),
                        ("Answer Correctness, answered rows (1-5)", f"{ci([corr[k] for k in ans])[0]:.2f} ({ci([corr[k] for k in ans])[1]:.2f} to {ci([corr[k] for k in ans])[2]:.2f})"),
                        ("Score (good - wrong) / rows", f"{sc[0]:+.3f} ({sc[1]:+.3f} to {sc[2]:+.3f})"),
                        ("Latency per request, seconds", f"median {np.median(secs):.0f}, 90th pct {np.percentile(secs, 90):.0f}, max {max(secs):.0f}"),
                        ("Rows missing a grade from some judge", str(len(miss)))]
        wrong = [k for k in ans if not good[k]]
        print(label, "wrong rows:", wrong)
    print(f"\n{'Metric':48s} | {'Wix (200)':40s} | TechQA (200)")
    for (m, w), (_, t) in zip(table["Wix (200)"], table["TechQA (200)"]):
        print(f"{m:48s} | {w:40s} | {t}")


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "judge": lambda: judge(sys.argv[2]), "report": report}[cmd]()
