"""Full offline comparison: ministral-14b vs mistral-large-4 on every saved context, same prompt, same judges.

usage (from eval/, keys loaded; DB = copy of the n8n database):
  python results/model_speed/full_run.py contexts DB     builds work_full/contexts.json: Wix first 100 + Wix fresh 100 (whole-article contexts taken from DB) and TechQA 100 (chunk contexts from answer_fix/work/inputs.jsonl)
  python results/model_speed/full_run.py gen A|ML4       resumable; A = ministral-14b-2512 via Mistral, ML4 = mistralai/mistral-large-4-1t-a52b (reasoning_effort low) via NVIDIA, 4 requests at once
  python results/model_speed/full_run.py judge A|ML4 mistral|nemotron|glm45     resumable, retries failures
  python results/model_speed/full_run.py report
Same prompt as the live agent (Wix or TechQA system message + JSON schema), temperature 0, one shot; sent = answered, confidence not low, needs_human_review not true.
Good = panel majority (Mistral, Nemotron, GLM-4.5) 4 or 5. Keys: "wix:<id>" and "techqa:<id>". Sets: wix_first (ids_wix.txt), wix_fresh (ids_wix_fresh.txt), techqa.
"""
import concurrent.futures as cf
import json
import os
import pathlib
import random
import re
import sqlite3
import sys
import time
from math import comb

import requests

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.nvidia_key import api_key as nvidia_key  # noqa: E402
RES = HERE.parent
P2 = RES / "prompts_v2"
WORK = HERE / "work_full"
WORK.mkdir(exist_ok=True)
sys.path.insert(0, str(P2))
sys.path.insert(0, str(RES / "fullarticle_copy"))
cmd = sys.argv[1]
_argv = sys.argv[:]
sys.argv = [sys.argv[0], "none"]
import test40 as T  # noqa: E402
import analyze_r10 as a  # noqa: E402
from harness.judges import score_answer_correctness  # noqa: E402
from harness.n8n_exec import node_outputs, unflatten  # noqa: E402

sys.argv = _argv
SETS = {"wix_first": ("wix", RES / "final_test/work/ids_wix.txt"), "wix_fresh": ("wix", RES / "fullarticle_copy/work/ids_wix_fresh.txt"), "techqa": ("techqa", RES / "final_test/work/ids_techqa.txt")}
NV = {"model": "mistralai/mistral-large-4-1t-a52b", "extra": {"reasoning_effort": "low"}}


def jl(p):
    p = pathlib.Path(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def contexts(db):
    wix = a.dataset("wix")
    tq = a.dataset("techqa")
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    by_q = {}
    for (d,) in c.execute("select d.data from execution_entity e join execution_data d on d.executionId=e.id where e.workflowId='tp513eed242ebb49' and e.status='success' and e.startedAt>='2026-11-04' order by e.id"):
        rd = unflatten(d)["resultData"]["runData"]
        q = next((p["body"]["text"] for p in node_outputs(rd, "Webhook") if isinstance(p.get("body"), dict) and "text" in p["body"]), None)
        try:
            j = rd["Root AI Agent"][0]["data"]["main"][0][0]["json"]
        except (KeyError, IndexError):
            continue
        seen, parts = set(), []
        for s in j.get("intermediateSteps") or []:
            if s["action"]["tool"].endswith("Retriever"):
                obs = s["observation"] if isinstance(s["observation"], str) else json.dumps(s["observation"])
                for src, text in T.docs_of(obs):
                    if (src, text[:100]) not in seen:
                        seen.add((src, text[:100]))
                        parts.append(f"[{src}]\n{text}")
        by_q[q] = "\n\n---\n\n".join(parts)[:60000]
    out, missing = {}, []
    wsys = T.live_system()
    for tag in ("wix_first", "wix_fresh"):
        for r in SETS[tag][1].read_text().split(","):
            q = wix[r]["question"]
            ctx = by_q.get(q) or next((v for k, v in by_q.items() if k and k.strip()[:80] == q.strip()[:80]), "")
            out[f"wix:{r}"] = {"set": tag, "question": q, "reference": wix[r]["expected_answer"], "context": ctx, "system": wsys}
            if not ctx:
                missing.append(r)
    for r in jl(RES / "answer_fix/work/inputs.jsonl"):
        if r["ds"] == "techqa":
            out[f"techqa:{r['row_id']}"] = {"set": "techqa", "question": r["question"], "reference": tq[r["row_id"]]["expected_answer"], "context": r["passages"], "system": r["system"]}
    (WORK / "contexts.json").write_text(json.dumps(out), encoding="utf-8")
    print({s: sum(1 for v in out.values() if v["set"] == s) for s in SETS}, "| empty wix contexts:", missing, "| mean chars", sum(len(v["context"]) for v in out.values()) // len(out))


def user_msg(c):
    return f'Question:\n{c["question"]}\n\nSearch results (the documentation passages found for this question):\n{c["context"]}'


def parse(raw, tokens, secs):
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except ValueError:
        d = {}
    sent = bool(d.get("answer")) and d.get("confidence") != "low" and d.get("needs_human_review") is not True
    return {"answer": d.get("answer") or "", "sent": sent, "conf": d.get("confidence"), "tokens": tokens, "secs": secs}


def one(arm, key, c):
    t0 = time.monotonic()
    system = c["system"] + "\n\n" + T.SCHEMA
    if arm == "A":
        raw, tok = T.chat(system, user_msg(c), json_mode=True)
        r = parse(raw, tok, round(time.monotonic() - t0, 1))
    else:
        body = {"model": NV["model"], "temperature": 0, "max_tokens": 3000, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg(c)}], **NV["extra"]}
        r = parse("", 0, 0)
        for attempt in range(5):
            try:
                resp = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers={"Authorization": "Bearer " + nvidia_key()}, json=body, timeout=150)
                if resp.status_code == 200:
                    d = resp.json()
                    r = parse(d["choices"][0]["message"].get("content") or "", d.get("usage", {}).get("total_tokens", 0), round(time.monotonic() - t0, 1))
                    break
                print("  HTTP", resp.status_code, resp.text[:70].replace("\n", " "), flush=True)
            except requests.RequestException as e:
                print("  ", type(e).__name__, flush=True)
            time.sleep(8 * (attempt + 1))
    r["key"] = key
    return r


def gen(arm):
    ctx = json.loads((WORK / "contexts.json").read_text(encoding="utf-8"))
    out = WORK / f"gen_{arm}.jsonl"
    rows = jl(out)
    done = {r["key"] for r in rows if r.get("tokens")}
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r.get("tokens")), encoding="utf-8")
    todo = [(k, c) for k, c in ctx.items() if k not in done]
    workers = int(os.environ.get("ML4_WORKERS", "4")) if arm == "ML4" else 1
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(lambda kc: (time.sleep(1.0) if arm == "A" else None) or one(arm, kc[0], kc[1]), todo):
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(arm, r["key"], "sent" if r["sent"] else "hand-off", r["conf"], "secs", r["secs"], "tokens", r["tokens"], flush=True)
    left = [r["key"] for r in jl(out) if not r.get("tokens")]
    print("gen", arm, "done; rows without an answer after retries:", left, flush=True)


def judge(arm, name):
    b = a.backend(name)
    ctx = json.loads((WORK / "contexts.json").read_text(encoding="utf-8"))
    out = WORK / f"judge_{name}_{arm}.jsonl"
    gens = {r["key"]: r for r in jl(WORK / f"gen_{arm}.jsonl")}
    for attempt in range(4):
        kept = [r for r in jl(out) if r["score"] is not None]
        out.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
        done = {r["key"] for r in kept}
        todo = [k for k in gens if k not in done]
        if not todo:
            break
        for k in todo:
            try:
                score, err = score_answer_correctness(b, ctx[k]["question"], ctx[k]["reference"], gens[k]["answer"] or "(no answer)"), None
            except Exception as e:  # noqa: BLE001
                score, err = None, f"{type(e).__name__}: {e}"[:150]
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": k, "score": score, "error": err}) + "\n")
            time.sleep({"mistral": 1.2, "nemotron": 0.4}.get(name, 0.8))
        print("judge", name, arm, "pass", attempt + 1, flush=True)
        time.sleep(20)
    print("judge", name, arm, "complete", len([r for r in jl(out) if r["score"] is not None]), flush=True)


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def boot(d, n=5000):
    rnd = random.Random(0)
    m = sorted(sum(d[rnd.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return m[int(.025 * n)], m[int(.975 * n)]


def report():
    ctx = json.loads((WORK / "contexts.json").read_text(encoding="utf-8"))
    G = {arm: {r["key"]: r for r in jl(WORK / f"gen_{arm}.jsonl")} for arm in ("A", "ML4")}
    S = {arm: {j: {r["key"]: r["score"] for r in jl(WORK / f"judge_{j}_{arm}.jsonl") if r["score"] is not None} for j in ("mistral", "nemotron", "glm45")} for arm in G}
    good = lambda arm, k: sum((S[arm][j].get(k) or 0) >= 4 for j in S[arm]) >= 2  # noqa: E731
    print("completeness:", {arm: (len(G[arm]), {j: len(s) for j, s in S[arm].items()}, "no-answer rows", sum(not r["answer"] for r in G[arm].values())) for arm in G})
    names = {"A": "ministral-14b", "ML4": "mistral-large-4"}
    sets = [(s, [k for k, v in ctx.items() if v["set"] == s]) for s in SETS] + [("wix_all", [k for k, v in ctx.items() if v["set"].startswith("wix")])]
    for s, keys in sets:
        print(f"\n=== {s} ({len(keys)} questions) ===")
        print("model | good sent | wrong sent | handed off | accuracy when answering | score (good - wrong)/N | good sent per judge M/N/G | mean tokens | mean secs")
        per = {}
        for arm in ("A", "ML4"):
            gs = [int(G[arm][k]["sent"] and good(arm, k)) for k in keys]
            ws = [int(G[arm][k]["sent"] and not good(arm, k)) for k in keys]
            per[arm] = (gs, ws)
            pj = "/".join(str(sum(G[arm][k]["sent"] and (S[arm][j].get(k) or 0) >= 4 for k in keys)) for j in ("mistral", "nemotron", "glm45"))
            secs = [G[arm][k]["secs"] for k in keys if G[arm][k].get("secs")]
            print(f"{names[arm]} | {sum(gs)} | {sum(ws)} | {len(keys) - sum(gs) - sum(ws)} | {100 * sum(gs) / max(1, sum(gs) + sum(ws)):.0f}% | {(sum(gs) - sum(ws)) / len(keys):.2f} | {pj} | {sum(G[arm][k]['tokens'] for k in keys) / len(keys):.0f} | {sum(secs) / max(1, len(secs)):.0f}")
        d = [(g2 - w2) - (g1 - w1) for g1, w1, g2, w2 in zip(per["A"][0], per["A"][1], per["ML4"][0], per["ML4"][1])]
        lo, hi = boot(d)
        gain = sum(1 for x, y in zip(per["A"][0], per["ML4"][0]) if y and not x)
        loss = sum(1 for x, y in zip(per["A"][0], per["ML4"][0]) if x and not y)
        wfix = sum(1 for x, y in zip(per["A"][1], per["ML4"][1]) if x and not y)
        wnew = sum(1 for x, y in zip(per["A"][1], per["ML4"][1]) if y and not x)
        print(f"  score change {sum(d) / len(d):+.2f} (95% range {lo:+.2f} to {hi:+.2f}) | good sent: +{gain} / -{loss}, McNemar p = {mcnemar(gain, loss):.3f} | wrong sent: {wfix} removed, {wnew} added")


if __name__ == "__main__":
    {"contexts": lambda: contexts(sys.argv[2]), "gen": lambda: gen(sys.argv[2]), "judge": lambda: judge(sys.argv[2], sys.argv[3]), "report": report}[cmd]()
