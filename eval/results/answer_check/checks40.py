"""Helpers used by the live evaluation (`pg`, the good/wrong rule; `score_answer_correctness`; the judge backends), and a 40-row test of four ways to catch wrong answers before sending. In the test everything is written and checked by mistral-large-4 (NVIDIA).

usage (from eval/, NVIDIA key loaded; judges need MISTRAL/ZAI keys too):
  python results/answer_check/checks40.py select        fixes the 40 rows and the approach-2 trigger (writes work/ids40.json, work/trigger.json, sha256 files) before any model call
  python results/answer_check/checks40.py run           runs approaches 1, 3, 4 (resumable; work/runs.json)
  python results/answer_check/checks40.py judge NAME    NAME = mistral|nemotron|glm45: grades the approach-1 drafts (resumable)
  python results/answer_check/checks40.py report
Rows: the 23 wrong answers sent by the new search + mistral-large-4 (arm R2 of ../embedding_test) and 17 good ones drawn at random (seed 0). Contexts, drafts and grades are the saved ones.
Approach 1: self-check fields in the same call (parts_not_covered, assumption_conflict, wrong_scope); hand off if any is set, or confidence low, or needs_human_review.
Approach 3: a second call on every answer: question + the articles + the draft; verdict PASS or HAND_OFF.
Approach 4: a second call on every answer: question + the draft only; verdict PASS or HAND_OFF on whether every part is answered.
Approach 2: the approach-3 check, but only for answers flagged by a free rule fixed in advance (confidence medium, or search margin or answer length in the lowest 10% of the 272 sent answers).
"""
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import re
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent
EMB = RES / "embedding_test"
WORK = HERE / "work"
WORK.mkdir(exist_ok=True)
sys.path.insert(0, str(RES.parent))
sys.path.insert(0, str(RES / "fullarticle_copy"))
from harness.nvidia_key import api_key  # noqa: E402
import requests  # noqa: E402

cmd = sys.argv[1]
_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
import analyze_r10 as a  # noqa: E402
from harness.judges import score_answer_correctness  # noqa: E402

sys.argv = _argv
JUDGES = ("mistral", "nemotron", "glm45")
SELF_CHECK = """

Before you answer, check your own answer against the customer's question and the articles. Add these three fields to the JSON:
"parts_not_covered": a list with every part of the customer's question that your answer does not fully address, or that the articles do not cover (an empty list if none);
"assumption_conflict": true if the question assumes something the articles contradict or limit, otherwise false;
"wrong_scope": true if the articles you used are about a different product, platform, edition or cause than the customer describes, otherwise false."""
SCHEMA1 = ('Return only a JSON object with the keys: answer (string), confidence ("high", "medium" or "low"), needs_handoff (boolean), needs_human_review (boolean), cited_source (string), '
           'parts_not_covered (list of strings), assumption_conflict (boolean), wrong_scope (boolean). No text outside the JSON.')
CHECK_WITH = """You check a draft support reply against the customer's question and the documentation passages. Return only a JSON object with the keys:
"missing_parts": list of the parts of the question the draft does not address;
"unsupported_claims": list of statements in the draft that the passages do not support;
"wrong_scope": true if the draft relies on passages about a different product, platform, edition or cause than the customer describes;
"assumption_conflict": true if the customer assumes something the passages contradict or limit and the draft does not say so;
"verdict": "PASS" if the draft answers every part from the passages, otherwise "HAND_OFF"."""
CHECK_WITHOUT = """You check whether a draft support reply fully answers the customer's question. You do not have the documentation; judge only whether every part of the question is addressed and whether the reply is a complete, direct answer. Return only a JSON object with the keys:
"parts": a list of {"part": the separate things the customer asks, "answered": true or false};
"verdict": "PASS" if every part is answered, otherwise "HAND_OFF"."""


def jl(p):
    p = pathlib.Path(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def pg(scores):
    return sum((s or 0) >= 4 for s in scores) >= 2


def load_base():
    g = {r["key"]: r for r in jl(EMB / "work/e2e_gen_R2.jsonl") if r.get("tokens")}
    s = {j: {r["key"]: r["score"] for r in jl(EMB / f"work/e2e_judge_{j}_R2.jsonl") if r["score"] is not None} for j in JUDGES}
    ctx = json.loads((EMB / "work/e2e_contexts.json").read_text(encoding="utf-8"))["R2"]
    return g, s, ctx


def retrieval_features(keys):
    spec = importlib.util.spec_from_file_location("queries", EMB / "queries.py")
    qm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qm)
    q = qm.questions()
    qn = json.loads((EMB / "work/q_nemo.json").read_text(encoding="utf-8"))
    chunks = {ds: json.loads((EMB / f"work/chunks_{ds}.json").read_text(encoding="utf-8")) for ds in ("wix", "techqa")}
    unit = lambda x: np.asarray(x, dtype=np.float32) / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)  # noqa: E731
    mats = {ds: unit(np.load(EMB / f"work/nemo_{ds}_title.npy")[:, :2048]) for ds in chunks}
    out = {}
    for k in keys:
        ds = q[k]["ds"]
        s = mats[ds] @ unit(np.array(qn[f"{k}|question"])[:2048])
        order = np.argsort(-s)
        first = order[0]
        second = next((i for i in order[1:60] if chunks[ds][i]["source"] != chunks[ds][first]["source"]), order[1])
        out[k] = float(s[first] - s[second])
    return out


def select():
    g, s, ctx = load_base()
    sent = [k for k in ctx if k in g and g[k]["sent"]]
    good_k = [k for k in sent if pg([s[j].get(k) for j in JUDGES])]
    wrong = sorted([k for k in sent if k not in good_k], key=lambda x: (x.split(":")[0], int(x.split(":")[1])))
    rnd = random.Random(0)
    good = sorted(rnd.sample(sorted(good_k), 17), key=lambda x: (x.split(":")[0], int(x.split(":")[1])))
    assert len(wrong) == 23 and len(good) == 17 and not set(wrong) & set(good)
    ids = {"wrong": wrong, "good": good}
    (WORK / "ids40.json").write_text(json.dumps(ids), encoding="utf-8")
    (WORK / "ids40.sha256").write_text(hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest(), encoding="utf-8")
    margin = retrieval_features(sent)
    lens = {k: len(g[k]["answer"]) for k in sent}
    m10 = float(np.percentile(list(margin.values()), 10))
    l10 = float(np.percentile(list(lens.values()), 10))
    flagged = {k: (g[k]["conf"] != "high") or margin[k] <= m10 or lens[k] <= l10 for k in sent}
    trig = {"margin_p10": m10, "length_p10": l10, "rule": "confidence medium OR margin <= p10 OR answer length <= p10 (percentiles over the 272 sent answers)",
            "flagged_share_of_272": sum(flagged.values()) / len(sent), "flagged": {k: bool(flagged[k]) for k in wrong + good}}
    (WORK / "trigger.json").write_text(json.dumps(trig), encoding="utf-8")
    (WORK / "trigger.sha256").write_text(hashlib.sha256(json.dumps(trig, sort_keys=True).encode()).hexdigest(), encoding="utf-8")
    print("wrong 23:", wrong)
    print("good 17:", good)
    print(f"trigger: margin <= {m10:.4f} or answer length <= {l10:.0f} chars or confidence medium -> flags {100 * trig['flagged_share_of_272']:.0f}% of all 272 sent answers; of the 40 rows: wrong flagged {sum(trig['flagged'][k] for k in wrong)}/23, good flagged {sum(trig['flagged'][k] for k in good)}/17")


def ml4(system, user, max_tokens=4000):
    body = {"model": "mistralai/mistral-large-4-1t-a52b", "temperature": 0, "max_tokens": max_tokens, "reasoning_effort": "low", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    for attempt in range(6):
        try:
            r = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers={"Authorization": "Bearer " + api_key()}, json=body, timeout=300)
            if r.status_code == 200:
                d = r.json()
                return d["choices"][0]["message"].get("content") or "", d.get("usage", {}).get("total_tokens", 0)
        except requests.RequestException:
            pass
        time.sleep(6 * (attempt + 1))
    return "", 0


def parse(raw):
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except ValueError:
        return {}


def run():
    g, s, ctx = load_base()
    ids = json.loads((WORK / "ids40.json").read_text(encoding="utf-8"))
    keys = ids["wrong"] + ids["good"]
    out_p = WORK / "runs.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}

    def job(task):
        approach, k = task
        c = ctx[k]
        if approach == "A1":
            raw, tok = ml4(c["system"] + SELF_CHECK + "\n\n" + SCHEMA1, f'Question:\n{c["question"]}\n\nSearch results (the documentation passages found for this question):\n{c["context"]}')
            d = parse(raw)
            return task, {"ok": bool(d.get("answer")), "answer": d.get("answer") or "", "conf": d.get("confidence"), "nhr": d.get("needs_human_review"), "parts_not_covered": d.get("parts_not_covered") or [],
                          "assumption_conflict": bool(d.get("assumption_conflict")), "wrong_scope": bool(d.get("wrong_scope")), "tokens": tok}
        if approach == "A3":
            raw, tok = ml4(CHECK_WITH, f'Question:\n{c["question"]}\n\nDocumentation passages:\n{c["context"]}\n\nDraft reply:\n{g[k]["answer"]}', 3000)
        else:
            raw, tok = ml4(CHECK_WITHOUT, f'Question:\n{c["question"]}\n\nDraft reply:\n{g[k]["answer"]}', 3000)
        d = parse(raw)
        return task, {"ok": d.get("verdict") in ("PASS", "HAND_OFF"), "verdict": d.get("verdict"), "detail": {kk: d.get(kk) for kk in ("missing_parts", "unsupported_claims", "wrong_scope", "assumption_conflict", "parts") if kk in d}, "tokens": tok}

    todo = [(a_, k) for a_ in ("A1", "A3", "A4") for k in keys if not out.get(f"{a_}|{k}", {}).get("ok")]
    print("calls to make:", len(todo), flush=True)
    with cf.ThreadPoolExecutor(int(os.environ.get("CHK_WORKERS", "6"))) as ex:
        for (a_, k), res in ex.map(job, todo):
            out[f"{a_}|{k}"] = res
            out_p.write_text(json.dumps(out), encoding="utf-8")
    bad = [kk for kk, v in out.items() if not v.get("ok")]
    print("done; calls without a usable result:", bad, flush=True)


def judge(name):
    b = a.backend(name)
    g, s, ctx = load_base()
    runs = json.loads((WORK / "runs.json").read_text(encoding="utf-8"))
    out = WORK / f"judge_{name}_A1.jsonl"
    for attempt in range(4):
        kept = [r for r in jl(out) if r["score"] is not None]
        out.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
        done = {r["key"] for r in kept}
        todo = [k.split("|", 1)[1] for k, v in runs.items() if k.startswith("A1|") and v.get("ok") and k.split("|", 1)[1] not in done]
        if not todo:
            break
        for k in todo:
            try:
                score, err = score_answer_correctness(b, ctx[k]["question"], ctx[k]["reference"], runs[f"A1|{k}"]["answer"]), None
            except Exception as e:  # noqa: BLE001
                score, err = None, type(e).__name__
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": k, "score": score, "error": err}) + "\n")
            time.sleep({"mistral": 1.2, "nemotron": 0.4}.get(name, 0.8))
        time.sleep(20)
    print("judge", name, "complete", len([r for r in jl(out) if r["score"] is not None]), flush=True)


def report():
    g, s, ctx = load_base()
    ids = json.loads((WORK / "ids40.json").read_text(encoding="utf-8"))
    trig = json.loads((WORK / "trigger.json").read_text(encoding="utf-8"))
    runs = json.loads((WORK / "runs.json").read_text(encoding="utf-8"))
    j1 = {j: {r["key"]: r["score"] for r in jl(WORK / f"judge_{j}_A1.jsonl") if r["score"] is not None} for j in JUDGES}
    base_tokens = {k: g[k]["tokens"] for k in ids["wrong"] + ids["good"]}
    print("completeness:", {"A1 ok": sum(1 for k, v in runs.items() if k.startswith("A1|") and v["ok"]), "A3 ok": sum(1 for k, v in runs.items() if k.startswith("A3|") and v["ok"]), "A4 ok": sum(1 for k, v in runs.items() if k.startswith("A4|") and v["ok"]),
                      "A1 grades": {j: len(x) for j, x in j1.items()}})
    results = {}
    for k in ids["wrong"] + ids["good"]:
        r1, r3, r4 = runs.get(f"A1|{k}", {}), runs.get(f"A3|{k}", {}), runs.get(f"A4|{k}", {})
        a1_hand = (not r1.get("answer")) or r1.get("conf") == "low" or r1.get("nhr") is True or bool(r1.get("parts_not_covered")) or r1.get("assumption_conflict") or r1.get("wrong_scope")
        a1_good = pg([j1[j].get(k) for j in JUDGES]) if r1.get("answer") else False
        results[k] = {"A0": "sent", "A1": "hand" if a1_hand else ("good" if a1_good else "wrong"),
                      "A3": "hand" if r3.get("verdict") == "HAND_OFF" else "sent", "A4": "hand" if r4.get("verdict") == "HAND_OFF" else "sent",
                      "A2": ("hand" if r3.get("verdict") == "HAND_OFF" else "sent") if trig["flagged"][k] else "sent",
                      "tok": {"A1": r1.get("tokens", 0) - base_tokens[k], "A3": r3.get("tokens", 0), "A4": r4.get("tokens", 0), "A2": r3.get("tokens", 0) if trig["flagged"][k] else 0}}

    def line(label, key, group):
        rows = ids[group]
        if key == "A1":
            handed = sum(results[k]["A1"] == "hand" for k in rows)
            ok_sent = sum(results[k]["A1"] == "good" for k in rows)
            wrong_sent = sum(results[k]["A1"] == "wrong" for k in rows)
            return handed, ok_sent, wrong_sent
        handed = sum(results[k][key] == "hand" for k in rows)
        base_good = group == "good"
        return handed, (len(rows) - handed) if base_good else 0, (len(rows) - handed) if not base_good else 0

    print("\napproach | of the 23 wrong: handed off or fixed / still wrong and sent | of the 17 good: still good and sent / lost (handed off or became wrong) | extra tokens per row (mean of 40) | extra calls per row")
    names = {"A0": "0 nothing (today)", "A1": "1 self-check inside the answer call", "A2": "2 checker on flagged answers only", "A3": "3 checker on every answer, reads the articles", "A4": "4 checker on every answer, question + answer only"}
    for key in ("A0", "A1", "A2", "A3", "A4"):
        if key == "A0":
            caught, still, kept, lost, tok, calls = 0, 23, 17, 0, 0.0, 0.0
        elif key == "A1":
            caught = sum(results[k]["A1"] in ("hand", "good") for k in ids["wrong"])
            still = 23 - caught
            kept = sum(results[k]["A1"] == "good" for k in ids["good"]) if False else sum(results[k]["A1"] in ("good",) for k in ids["good"])
            lost = 17 - kept
            tok = float(np.mean([results[k]["tok"]["A1"] for k in ids["wrong"] + ids["good"]]))
            calls = 0.0
        else:
            caught = sum(results[k][key] == "hand" for k in ids["wrong"])
            still = 23 - caught
            lost = sum(results[k][key] == "hand" for k in ids["good"])
            kept = 17 - lost
            tok = float(np.mean([results[k]["tok"][key] for k in ids["wrong"] + ids["good"]]))
            calls = (sum(trig["flagged"][k] for k in ids["wrong"] + ids["good"]) / 40) if key == "A2" else 1.0
        print(f"{names[key]:52s} | {caught:2d} / {still:2d} | {kept:2d} / {lost:2d} | {tok:6.0f} | {calls:.2f}")
    mean_base = float(np.mean(list(base_tokens.values())))
    print(f"\nreference: the answer call itself uses about {mean_base:.0f} tokens on these 40 rows.")
    print(f"approach 2 trigger flags {100 * trig['flagged_share_of_272']:.0f}% of all 272 sent answers (on these 40 rows: wrong {sum(trig['flagged'][k] for k in ids['wrong'])}/23, good {sum(trig['flagged'][k] for k in ids['good'])}/17).")
    print("\nwhich wrong rows each approach caught (hand-off or fixed):")
    for key in ("A1", "A3", "A4", "A2"):
        ok = [k for k in ids["wrong"] if (results[k]["A1"] in ("hand", "good") if key == "A1" else results[k][key] == "hand")]
        print(f"  {names[key]}: {ok}")
    print("\ngood rows lost:")
    for key in ("A1", "A3", "A4", "A2"):
        lost = [k for k in ids["good"] if (results[k]["A1"] != "good" if key == "A1" else results[k][key] == "hand")]
        print(f"  {names[key]}: {lost}")
    (WORK / "results.json").write_text(json.dumps(results), encoding="utf-8")


if __name__ == "__main__":
    {"select": select, "run": run, "judge": lambda: judge(sys.argv[2]), "report": report}[cmd]()
