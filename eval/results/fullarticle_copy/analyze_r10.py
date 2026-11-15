"""Score the -r10 (whole articles) run against the live system's saved run on the same 100 held-out questions per dataset.

usage (from eval/, keys loaded): python results/fullarticle_copy/analyze_r10.py runs DB            (no API: reads the n8n database copy; lists retrieval problems and tokens)
                                 python results/fullarticle_copy/analyze_r10.py grade JUDGE        (JUDGE = mistral|nemotron|glm45; resumable, retries failures)
                                 python results/fullarticle_copy/analyze_r10.py report [j1+j2+j3]  (default panel: mistral+nemotron+glm45, majority)
Baseline = the live system's recorded run (rerun_fixed, decision in fixes_round3/runs/replay_{ds}.jsonl), graded by the same judges
(Mistral labels_{ds}.jsonl, Nemotron judge2_{ds}_old.jsonl, GLM-4.5 judge_panel/v2_glm45_{ds}.jsonl). Good = correctness >= 4.
Sent = status "answered". Rows whose n8n execution shows a retrieval problem (Cohere 429, empty search) are invalid and must be re-run, not scored.
"""
import json
import pathlib
import random
import sqlite3
import sys
import time
from math import comb

HERE = pathlib.Path(__file__).resolve().parent
W = HERE / "work"
RES = HERE.parent
EVAL = RES.parent
sys.path.insert(0, str(EVAL))
from harness.backends import MistralChat, OpenAICompatChat  # noqa: E402
from harness.judges import score_answer_correctness  # noqa: E402
from harness.n8n_exec import node_outputs, unflatten  # noqa: E402

DS = ("wix", "techqa")
COPY_ID = {"wix": "tp513eed242ebb49", "techqa": "tp41367be8f80c44"}


def jl(p):
    p = pathlib.Path(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def responses(ds):
    """Return the run-file rows with the webhook reply (status, answer) lifted to the top level; the last record per row wins."""
    out = {}
    for r in jl(W / f"r10_{ds}.jsonl"):
        out[str(r["row_id"])] = {"row_id": str(r["row_id"]), **(r.get("response") or {})}
    return out


def dataset(ds):
    return {str(r["id"]): r for r in json.loads((EVAL / f"data/{ds}_full.json").read_text(encoding="utf-8"))}


def baseline(ds):
    sent = {str(r["row_id"]): r["old_status"] == "answered" for r in jl(RES / f"fixes_round3/runs/replay_{ds}.jsonl")}
    g = {"mistral": {str(r["row_id"]): r["correctness"] for r in jl(RES / f"fixes_round3/runs/labels_{ds}.jsonl")},
         "nemotron": {str(r["row_id"]): r["correctness"] for r in jl(RES / f"eval_quality/judge2_{ds}_old.jsonl")},
         "glm45": {r["row_id"]: r["score"] for r in jl(RES / f"judge_panel/v2_glm45_{ds}.jsonl") if r["score"] is not None}}
    return sent, g


def runs(db):
    """Return, per r10 execution: question, retrieval problem, estimated tokens, seconds. The last execution per question wins."""
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = {}
    for ds in DS:
        by_q = {}
        for eid, started, stopped, data in c.execute("select e.id,e.startedAt,e.stoppedAt,d.data from execution_entity e join execution_data d on d.executionId=e.id "
                                                     "where e.workflowId=? and e.status='success' order by e.id", (COPY_ID[ds],)):
            rd = unflatten(data)["resultData"]["runData"]
            q = next((p["body"]["text"] for p in node_outputs(rd, "Webhook") if isinstance(p.get("body"), dict) and "text" in p["body"]), None)
            prob = next(iter(o.get("retrieval_problem") for o in node_outputs(rd, "Check Retrieval")), "n/a") if "Check Retrieval" in rd else "n/a"
            tok = 0
            for node, rs in rd.items():
                for run in rs:
                    for items in (run.get("data", {}).get("ai_languageModel") or []):
                        for it in items or []:
                            tok += (it["json"].get("tokenUsageEstimate") or {}).get("totalTokens", 0)
            secs = None
            if started and stopped:
                from datetime import datetime
                f = "%Y-%m-%d %H:%M:%S.%f"
                secs = (datetime.strptime(stopped, f) - datetime.strptime(started, f)).total_seconds()
            by_q[q] = {"exec": eid, "problem": prob, "tokens": tok, "secs": secs}
        data = dataset(ds)
        rows = {str(r["row_id"]): r for r in jl(W / f"r10_{ds}.jsonl")}
        res = {}
        for rid in rows:
            q = data[rid]["question"]
            hit = by_q.get(q) or next((v for k, v in by_q.items() if k and k.strip()[:80] == q.strip()[:80]), None)
            res[rid] = hit
        out[ds] = res
        bad = {r: v for r, v in res.items() if v is None or v["problem"] not in (None, "n/a")}
        toks = [v["tokens"] for v in res.values() if v and v["problem"] is None]
        secs = [v["secs"] for v in res.values() if v and v["problem"] is None and v["secs"]]
        print(f"{ds}: {len(rows)} responses, {len(res) - len(bad)} clean executions; retrieval problems / missing: {sorted(bad, key=int)} "
              f"{ {r: (v or {}).get('problem') for r, v in bad.items()} }")
        if toks:
            print(f"   mean estimated tokens per question {sum(toks) / len(toks):.0f}, mean seconds {sum(secs) / max(1, len(secs)):.0f}")
    (W / "r10_runs.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def backend(name):
    if name == "mistral":
        return MistralChat("ministral-14b-2512")
    if name == "nemotron":
        return OpenAICompatChat("nvidia/nemotron-3.5-super-vl-120b-a12b", "NVIDIA_API_KEY", "https://integrate.api.nvidia.com/v1",
                                extra_body={"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 400})
    return OpenAICompatChat("glm-4.5-flash", "ZAI_API_KEY", "https://api.z.ai/api/paas/v4", extra_body={"thinking": {"type": "disabled"}, "max_tokens": 400})


def grade(name):
    b = backend(name)
    for ds in DS:
        data, out = dataset(ds), W / f"r10_judge_{name}_{ds}.jsonl"
        for attempt in range(4):
            kept = [r for r in jl(out) if r["score"] is not None]
            out.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
            done = {r["row_id"] for r in kept}
            todo = [r for r in responses(ds).values() if r.get("status") == "answered" and r["row_id"] not in done]
            if not todo:
                break
            for r in todo:
                rid = str(r["row_id"])
                try:
                    score, err = score_answer_correctness(b, data[rid]["question"], data[rid]["expected_answer"], r["answer"]), None
                except Exception as e:  # recorded as missing, retried on the next pass
                    score, err = None, f"{type(e).__name__}: {e}"[:200]
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"row_id": rid, "score": score, "error": err}) + "\n")
                time.sleep({"mistral": 1.5, "nemotron": 0.5}.get(name, 1.0))
            print("judge", name, ds, "pass", attempt + 1, flush=True)
            time.sleep(20)
        print("judge", name, ds, "complete:", len([r for r in jl(out) if r["score"] is not None]), flush=True)


def boot(new, old, n=5000):
    rnd, d = random.Random(0), [a - b for a, b in zip(new, old)]
    s = sorted(sum(d[rnd.randrange(len(d))] for _ in d) for _ in range(n))
    return f"{sum(d):+d} (range {s[int(.025 * n)]:+d} to {s[int(.975 * n)]:+d})"


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def report(panel):
    runs_info = json.loads((W / "r10_runs.json").read_text(encoding="utf-8")) if (W / "r10_runs.json").exists() else {}
    for ds in DS:
        recs = responses(ds)
        bsent, bg = baseline(ds)
        ids = sorted(bsent, key=int)
        new_g = {j: {r["row_id"]: r["score"] for r in jl(W / f"r10_judge_{j}_{ds}.jsonl") if r["score"] is not None} for j in ("mistral", "nemotron", "glm45")}
        nsent = {k: recs.get(k, {}).get("status") == "answered" for k in ids}
        missing = [k for k in ids if k not in recs]
        print(f"\n=== {ds}: baseline = live system's saved run; new = whole articles (-r10). Responses {len(recs)}/{len(ids)}; missing {missing[:10]} ===")
        ok_new = all(all(k in new_g[j] for k in ids if nsent[k]) for j in panel)
        if not ok_new:
            print("  grading incomplete for", [j for j in panel if not all(k in new_g[j] for k in ids if nsent[k])])
            continue
        gb = {k: sum((bg[j].get(k) or 0) >= 4 for j in panel) >= 2 for k in ids}
        gn = {k: sum((new_g[j].get(k) or 0) >= 4 for j in panel) >= 2 for k in ids}
        print("                   | sent | good sent | wrong sent | sent to a person | good sent by single judges: " + " / ".join(("mistral", "nemotron", "glm45")))
        for label, sent, g, per in (("live (before)", bsent, gb, bg), ("whole articles", nsent, gn, new_g)):
            cells = " / ".join(str(sum(sent[k] and (per[j].get(k) or 0) >= 4 for k in ids)) for j in ("mistral", "nemotron", "glm45"))
            print(f"  {label:17s}| {sum(sent.values()):4d} | {sum(sent[k] and g[k] for k in ids):9d} | {sum(sent[k] and not g[k] for k in ids):10d} | {len(ids) - sum(sent.values()):16d} | {cells}")
        gain = [k for k in ids if nsent[k] and gn[k] and not (bsent[k] and gb[k])]
        loss = [k for k in ids if bsent[k] and gb[k] and not (nsent[k] and gn[k])]
        fixed = [k for k in ids if bsent[k] and not gb[k] and not (nsent[k] and not gn[k])]
        broke = [k for k in ids if nsent[k] and not gn[k] and not (bsent[k] and not gb[k])]
        print(f"  good sent: gains {len(gain)} {gain}, loses {len(loss)} {loss}")
        print(f"  wrong sent: removes {len(fixed)} {fixed}, adds {len(broke)} {broke}")
        print(f"  paired bootstrap, good sent new - old (95%): {boot([int(nsent[k] and gn[k]) for k in ids], [int(bsent[k] and gb[k]) for k in ids])}; exact McNemar p = {mcnemar(len(gain), len(loss)):.3f}")
        print(f"  paired bootstrap, wrong sent new - old (95%): {boot([int(nsent[k] and not gn[k]) for k in ids], [int(bsent[k] and not gb[k]) for k in ids])}")
        info = runs_info.get(ds) or {}
        toks = [v["tokens"] for k, v in info.items() if v and v["problem"] is None]
        if toks:
            print(f"  estimated tokens per question: whole articles {sum(toks) / len(toks):.0f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "runs":
        runs(sys.argv[2])
    elif cmd == "grade":
        grade(sys.argv[2])
    else:
        report(tuple((sys.argv[2] if len(sys.argv) > 2 else "mistral+nemotron+glm45").split("+")))
