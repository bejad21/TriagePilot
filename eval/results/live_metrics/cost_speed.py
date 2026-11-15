"""Speed and token usage of the running workflows, read from n8n's own execution records (the saved input and output of every node of every request).
usage: copy database.sqlite, database.sqlite-wal and database.sqlite-shm out of the n8n container into one folder, then
  python results/live_metrics/cost_speed.py DB_FOLDER      -> prints the tables and writes cost_speed.md next to this file
Only successful executions that started at or after SINCE and reached the Answer node are used.
n8n stores execution data in `flatted` format (a list where every reference is the index of another entry); decode() unpacks it."""
import json
import pathlib
import sqlite3
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
DB = pathlib.Path(sys.argv[1]) / "database.sqlite"
WF = {"chZker1NOY9vZNSd": "Wix", "GpD9Tj3q1qwkqNxP": "TechQA"}
SINCE = "2026-11-07 21:00:00"
STAGES = {"Jailbreak Intent Check": "guard check on the question", "Embed Question": "embed question", "Search Qdrant": "vector search", "Answer": "answer call (mistral-large-4)",
          "Check Outbound Content": "guard check on the reply"}


def decode(text):
    arr = json.loads(text)
    memo = {}

    def get(i):
        if i in memo:
            return memo[i]
        v = arr[i]
        if isinstance(v, dict):
            out = {}
            memo[i] = out
            for k, x in v.items():
                out[k] = get(int(x)) if isinstance(x, str) else x
            return out
        if isinstance(v, list):
            out = []
            memo[i] = out
            for x in v:
                out.append(get(int(x)) if isinstance(x, str) else x)
            return out
        memo[i] = v
        return v

    return get(0)


def pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * len(s)))]


def stats(vals):
    return f"median {statistics.median(vals):.1f}, mean {statistics.fmean(vals):.1f}, 90th pct {pct(vals, .9):.1f}, max {max(vals):.1f}"


con = sqlite3.connect(DB)
res = {}
for wid, name in WF.items():
    rows = con.execute("select e.id, e.startedAt, e.stoppedAt from execution_entity e where e.workflowId=? and e.status='success' and e.startedAt>=? order by e.startedAt", (wid, SINCE)).fetchall()
    recs = []
    for eid, started, stopped in rows:
        data = con.execute("select data from execution_data where executionId=?", (eid,)).fetchone()[0]
        run = decode(data)["resultData"]["runData"]
        if "Answer" not in run:
            continue
        stage = {n: run[n][0].get("executionTime", 0) / 1000 for n in STAGES if n in run}
        try:
            usage = run["Answer"][0]["data"]["main"][0][0]["json"].get("usage") or {}
        except (KeyError, IndexError, TypeError):
            usage = {}
        t0 = min(r[0]["startTime"] for r in run.values())
        t1 = max(r[0]["startTime"] + r[0].get("executionTime", 0) for r in run.values())
        recs.append({"stage": stage, "usage": usage, "total": (t1 - t0) / 1000})
    res[name] = recs

lines = ["# Speed and token usage of the running workflows", "",
         "Read from n8n's execution records by `cost_speed.py`. Each row of a table is one successful request that reached the answer step (Wix and TechQA are separate). Times are in seconds.", ""]
for name, recs in res.items():
    lines += [f"## {name}: {len(recs)} requests", ""]
    lines += ["| Step | Seconds per request |", "|---|---|"]
    lines += [f"| Whole request | {stats([r['total'] for r in recs])} |"]
    for node, label in STAGES.items():
        v = [r["stage"][node] for r in recs if node in r["stage"]]
        if v:
            lines.append(f"| {label} | {stats(v)} |")
    rest = [r["total"] - r["stage"].get("Answer", 0) for r in recs]
    lines.append(f"| Everything except the answer call | {stats(rest)} |")
    pt = [r["usage"]["prompt_tokens"] for r in recs if r["usage"].get("prompt_tokens")]
    ct = [r["usage"]["completion_tokens"] for r in recs if r["usage"].get("completion_tokens")]
    lines += ["", "| Tokens per request | Input (prompt) | Output (completion) |", "|---|---|---|"]
    if pt and ct:
        lines.append(f"| Median | {statistics.median(pt):,.0f} | {statistics.median(ct):,.0f} |")
        lines.append(f"| Mean | {statistics.fmean(pt):,.0f} | {statistics.fmean(ct):,.0f} |")
        lines.append(f"| 90th percentile | {pct(pt, .9):,.0f} | {pct(ct, .9):,.0f} |")
        lines.append(f"| Requests with usage recorded | {len(pt)} of {len(recs)} | {len(ct)} of {len(recs)} |")
    else:
        lines.append("| usage not recorded | | |")
    lines.append("")
text = "\n".join(lines)
(HERE / "cost_speed.md").write_text(text, encoding="utf-8")
print(text)
