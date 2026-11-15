"""Offline test of the v2 prompts on 40 Wix questions: arm A (baseline prompt) against arm B (reader/writer/checker prompts), same saved contexts, same model.

usage (from eval/, keys loaded):
  python results/prompts_v2/test40.py select                (writes work/ids40.json and its sha256, before any run)
  python results/prompts_v2/test40.py contexts DB           (builds work/contexts40.json from the saved whole-article runs in the n8n database copy)
  python results/prompts_v2/test40.py gen A|B               (MISTRAL_API_KEY; resumable)
  python results/prompts_v2/test40.py judge A|B JUDGE       (JUDGE = mistral|nemotron|glm45; resumable, retries failures)
  python results/prompts_v2/test40.py report
The 40 = the 20 wrong answers the whole-article copy sent (tuning set and held-out set Wix questions) + 20 good ones at random (seed 0, 10 from each set).
Context = the unique documents the copy's searches returned, 60,000 characters at most, identical for both arms.
Arm A = the live Wix prompt, one shot, JSON output. Arm B = prompts_v2: reader, writer, checker, one rewrite if the checker says FIX.
Sent: A as in the live rule (parsed, confidence not low, needs_human_review not true); B unless the writer says LOW or the checker says HAND_OFF. Temperature 0, ministral-14b-2512.
Good = panel majority (Mistral, Nemotron, GLM-4.5) 4 or 5.
"""
import hashlib
import json
import os
import pathlib
import random
import re
import sqlite3
import sys
import time

import requests

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent
WORK = HERE / "work"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RES / "fullarticle_copy"))
sys.path.insert(0, str(RES.parent))
cmd = sys.argv[1]
_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
import analyze_r10 as a  # noqa: E402
import prompts_v2 as P  # noqa: E402
from harness.judges import score_answer_correctness  # noqa: E402
from harness.n8n_exec import node_outputs, unflatten  # noqa: E402

sys.argv = _argv
SCHEMA = ('Return only a JSON object with the keys: answer (string), confidence ("high", "medium" or "low"), needs_handoff (boolean), '
          'needs_human_review (boolean), cited_source (string). No text outside the JSON.')
LIVE_WIX = RES / "improve_round5/backup_live/chZker1NOY9vZNSd.live.json"
TAGS = {"36": "T8", "59": "T1", "111": "T4", "115": "T3", "121": "T4", "3": "T3", "5": "T6", "7": "T3", "144": "T6", "150": "T1", "168": "T6", "172": "T2/T5", "173": "T2",
        "186": "T1", "189": "T3", "190": "T1/T3/T8", "194": "T7", "197": "T3/T4/T8", "198": "T6", "199": "T6"}


def jl(p):
    p = pathlib.Path(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def pg(scores):
    return sum((s or 0) >= 4 for s in scores) >= 2


def select():
    lists = json.loads((WORK / "wrong_lists.json").read_text(encoding="utf-8"))
    wrong = sorted(set(lists["wix_first100_copy"]) | set(lists["wix_fresh100_copy"]), key=int)
    resp = a.responses("wix")
    gr = {j: {r["row_id"]: r["score"] for r in jl(RES / f"fullarticle_copy/work/r10_judge_{j}_wix.jsonl") if r["score"] is not None} for j in ("mistral", "nemotron", "glm45")}
    good_first = sorted([k for k, r in resp.items() if r.get("status") == "answered" and pg([gr[j].get(k) for j in gr])], key=int)
    fg = json.loads((RES / "fullarticle_copy/work/r10_wix_fresh_graded.json").read_text(encoding="utf-8"))
    good_fresh = sorted([k for k, v in fg.items() if v["sent"] and v["good"]], key=int)
    rnd = random.Random(0)
    good = sorted(rnd.sample(good_first, 10) + rnd.sample(good_fresh, 10), key=int)
    ids = {"wrong": wrong, "good": good}
    assert len(wrong) == 20 and len(good) == 20 and not set(wrong) & set(good)
    (WORK / "ids40.json").write_text(json.dumps(ids), encoding="utf-8")
    (WORK / "ids40.sha256").write_text(hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest(), encoding="utf-8")
    print("wrong", wrong, "\ngood", good)


def docs_of(obs):
    out = []
    try:
        parts = json.loads(obs)
    except ValueError:
        return out
    parts = parts if isinstance(parts, list) else [parts]
    for p in parts:
        if isinstance(p, dict) and isinstance(p.get("text"), str):
            try:
                p = json.loads(p["text"])
            except ValueError:
                continue
        if isinstance(p, dict):
            for d in p.get("documents", []):
                out.append(((d.get("metadata") or {}).get("source") or "unknown", d.get("pageContent") or ""))
    return out


def contexts(db):
    ids = json.loads((WORK / "ids40.json").read_text(encoding="utf-8"))
    allids = ids["wrong"] + ids["good"]
    data = a.dataset("wix")
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
                for src, text in docs_of(obs):
                    if (src, text[:100]) not in seen:
                        seen.add((src, text[:100]))
                        parts.append(f"[{src}]\n{text}")
        by_q[q] = "\n\n---\n\n".join(parts)[:60000]
    out, missing = {}, []
    for r in allids:
        q = data[r]["question"]
        ctx = by_q.get(q) or next((v for k, v in by_q.items() if k and k.strip()[:80] == q.strip()[:80]), "")
        out[r] = {"question": q, "reference": data[r]["expected_answer"], "context": ctx}
        if not ctx:
            missing.append(r)
    (WORK / "contexts40.json").write_text(json.dumps(out), encoding="utf-8")
    print("contexts", len(out), "empty:", missing, "| mean chars", sum(len(v["context"]) for v in out.values()) // len(out))


def chat(system, user, max_tokens=1800, json_mode=False):
    body = {"model": "ministral-14b-2512", "temperature": 0, "max_tokens": max_tokens, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    for attempt in range(6):
        try:
            r = requests.post("https://api.mistral.ai/v1/chat/completions", headers={"Authorization": "Bearer " + os.environ["MISTRAL_API_KEY"]}, json=body, timeout=240)
            if r.status_code == 200:
                d = r.json()
                return d["choices"][0]["message"]["content"] or "", d.get("usage", {}).get("total_tokens", 0)
            print("  HTTP", r.status_code, r.text[:90].replace("\n", " "), flush=True)
        except requests.RequestException as e:
            print("  ", type(e).__name__, flush=True)
        time.sleep(5 * (attempt + 1))
    return "", 0


def live_system():
    wf = json.loads(LIVE_WIX.read_text(encoding="utf-8"))
    wf = wf[0] if isinstance(wf, list) else wf
    msg = next(n for n in wf["nodes"] if n["name"] == "Root AI Agent")["parameters"]["options"]["systemMessage"]
    return re.sub(r"\{\{[^}]*\}\}", "2026-11-05", msg)


def run_A(c):
    raw, tok = chat(live_system() + "\n\n" + SCHEMA, f'Question:\n{c["question"]}\n\nSearch results (the documentation passages found for this question):\n{c["context"]}', json_mode=True)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except ValueError:
        d = {}
    sent = bool(d.get("answer")) and d.get("confidence") != "low" and d.get("needs_human_review") is not True
    return {"answer": d.get("answer") or "", "sent": sent, "conf": d.get("confidence"), "tokens": tok, "calls": 1}


def tail(draft):
    conf = (re.findall(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", draft, re.I) or [""])[-1].upper()
    body = re.split(r"\n\s*SOURCES:", draft)[0].strip()
    return body, conf


def run_B(c):
    dom = P.DOMAIN["wix"]
    notes, t1 = chat(P.READER.format(domain=dom), f'Question:\n{c["question"]}\n\nArticles:\n{c["context"]}', 1500)
    draft, t2 = chat(P.WRITER.format(domain=dom), f'Question:\n{c["question"]}\n\nNOTES:\n{notes}', 1500)
    check, t3 = chat(P.CHECKER, f'Question:\n{c["question"]}\n\nNOTES:\n{notes}\n\nDRAFT:\n{tail(draft)[0]}', 700)
    verdict = (re.findall(r"VERDICT:\s*(PASS|FIX|HAND_OFF)", check, re.I) or ["PASS"])[-1].upper()
    calls, tok, rewrote = 3, t1 + t2 + t3, False
    if verdict == "FIX":
        problems = check.split("VERDICT:")[-1].strip()
        draft2, t4 = chat(P.WRITER.format(domain=dom) + P.FIX_SUFFIX.format(problems=problems), f'Question:\n{c["question"]}\n\nNOTES:\n{notes}\n\nFIRST DRAFT:\n{tail(draft)[0]}', 1500)
        if draft2.strip():
            draft, rewrote = draft2, True
        calls, tok = 4, tok + t4
    body, conf = tail(draft)
    sent = bool(body) and conf != "LOW" and verdict != "HAND_OFF"
    return {"answer": body, "sent": sent, "conf": conf, "verdict": verdict, "rewrote": rewrote, "tokens": tok, "calls": calls, "notes": notes[:1500], "check": check[:600]}


def gen(arm):
    ctx = json.loads((WORK / "contexts40.json").read_text(encoding="utf-8"))
    out = WORK / f"gen_{arm}.jsonl"
    done = {r["id"] for r in jl(out) if r.get("tokens")}  # a failed API call leaves 0 tokens and is retried
    for rid, c in ctx.items():
        if rid in done:
            continue
        r = (run_A if arm == "A" else run_B)(c)
        r["id"] = rid
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(arm, rid, "sent" if r["sent"] else "hand-off", r.get("conf"), r.get("verdict", ""), "tokens", r["tokens"], flush=True)
        time.sleep(1.0)


def judge(arm, name):
    b = a.backend(name)
    ctx = json.loads((WORK / "contexts40.json").read_text(encoding="utf-8"))
    out = WORK / f"judge_{name}_{arm}.jsonl"
    last = {r["id"]: r for r in jl(WORK / f"gen_{arm}.jsonl")}
    for attempt in range(4):
        kept = [r for r in jl(out) if r["score"] is not None]
        out.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
        done = {r["id"] for r in kept}
        todo = [rid for rid in last if rid not in done]
        if not todo:
            break
        for rid in todo:
            try:
                score, err = score_answer_correctness(b, ctx[rid]["question"], ctx[rid]["reference"], last[rid]["answer"] or "(no answer)"), None
            except Exception as e:  # noqa: BLE001
                score, err = None, f"{type(e).__name__}: {e}"[:150]
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": rid, "score": score, "error": err}) + "\n")
            time.sleep({"mistral": 1.5, "nemotron": 0.5}.get(name, 1.0))
        print("judge", name, arm, "pass", attempt + 1, flush=True)
        time.sleep(20)
    print("judge", name, arm, "complete", len([r for r in jl(out) if r["score"] is not None]), flush=True)


def report():
    ids = json.loads((WORK / "ids40.json").read_text(encoding="utf-8"))
    arms = {}
    for arm in ("A", "B"):
        g = {r["id"]: r for r in jl(WORK / f"gen_{arm}.jsonl")}
        sc = {j: {r["id"]: r["score"] for r in jl(WORK / f"judge_{j}_{arm}.jsonl") if r["score"] is not None} for j in ("mistral", "nemotron", "glm45")}
        arms[arm] = (g, sc)
    print("check: drafts", {k: len(v[0]) for k, v in arms.items()}, "| grades", {k: {j: len(s) for j, s in v[1].items()} for k, v in arms.items()})
    def good(arm, rid):
        g, sc = arms[arm]
        return pg([sc[j].get(rid) for j in sc])
    for group in ("wrong", "good"):
        rows = ids[group]
        print(f"\n=== the 20 questions the whole-article copy got {'WRONG' if group == 'wrong' else 'RIGHT (control, checks nothing breaks)'} ===")
        print("arm | good and sent | wrong and sent | handed off | of the handed off: good drafts | mean tokens | mean calls")
        for arm in ("A", "B"):
            g, sc = arms[arm]
            gs = sum(g[r]["sent"] and good(arm, r) for r in rows)
            ws = sum(g[r]["sent"] and not good(arm, r) for r in rows)
            ho = [r for r in rows if not g[r]["sent"]]
            print(f"  {arm} | {gs} | {ws} | {len(ho)} | {sum(good(arm, r) for r in ho)} | {sum(g[r]['tokens'] for r in rows) / len(rows):.0f} | {sum(g[r]['calls'] for r in rows) / len(rows):.1f}")
        if group == "wrong":
            fixed = [r for r in rows if arms['B'][0][r]['sent'] and good('B', r)]
            print("  B good and sent on:", [(r, TAGS.get(r)) for r in fixed])
            ho = [r for r in rows if not arms['B'][0][r]['sent']]
            print("  B handed off:", [(r, TAGS.get(r), 'draft good' if good('B', r) else 'draft not good') for r in ho])
            still = [r for r in rows if arms['B'][0][r]['sent'] and not good('B', r)]
            print("  B still wrong and sent:", [(r, TAGS.get(r)) for r in still])
        else:
            broke = [r for r in rows if arms['A'][0][r]['sent'] and good('A', r) and not (arms['B'][0][r]['sent'] and good('B', r))]
            print("  good under A but not good-and-sent under B:", broke)
    print("\nall 40 together (CRAG-style score = (good sent - wrong sent) / 40):")
    for arm in ("A", "B"):
        g = arms[arm][0]
        allr = ids["wrong"] + ids["good"]
        gs = sum(g[r]["sent"] and good(arm, r) for r in allr)
        ws = sum(g[r]["sent"] and not good(arm, r) for r in allr)
        print(f"  {arm}: good sent {gs}, wrong sent {ws}, handed off {40 - gs - ws}, score {(gs - ws) / 40:.2f}")
    print("single judges, good and sent, B vs A on all 40:", {j: (sum(arms['A'][0][r]['sent'] and (arms['A'][1][j].get(r) or 0) >= 4 for r in ids['wrong'] + ids['good']), sum(arms['B'][0][r]['sent'] and (arms['B'][1][j].get(r) or 0) >= 4 for r in ids['wrong'] + ids['good'])) for j in ('mistral', 'nemotron', 'glm45')})


if __name__ == "__main__":
    {"select": select, "contexts": lambda: contexts(sys.argv[2]), "gen": lambda: gen(sys.argv[2]), "judge": lambda: judge(sys.argv[2], sys.argv[3]), "report": report}[cmd]()
