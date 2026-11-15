"""How well do the fast Mistral-API models grade compared with the judges already in the panel?
Rows: 60 answers of the offline best version that all three panel judges have graded (every wrong-by-panel answer among the 300, filled with good ones, seed 3).
A candidate is compared with (a) the panel majority (good = 2 of 3 give >= 4) and (b) the Mistral judge ministral-14b-2512, using its saved grades (that model is not called here).
usage (from eval/, MISTRAL_API_KEY loaded): python results/rerank_hybrid/judge_candidates.py MODEL [extra_json]
  e.g. judge_candidates.py voxtral-small-latest      |      judge_candidates.py mistral-large-4 '{"top_p":1,"reasoning_effort":"low"}'"""
import json
import os
import pathlib
import random
import re
import sys
import time

import requests

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness import prompts  # noqa: E402

E = HERE.parent / "embedding_test/work"
J = ("mistral", "nemotron", "glm45")


def jl(p):
    return [json.loads(x) for x in pathlib.Path(p).read_text(encoding="utf-8").splitlines() if x.strip()] if pathlib.Path(p).exists() else []


ctx = json.loads((E / "e2e_contexts.json").read_text(encoding="utf-8"))["R2"]
gen = {r["key"]: r for r in jl(E / "e2e_gen_R2.jsonl") if r.get("tokens") and r["sent"] and r["answer"].strip()}
sc = {j: {r["key"]: r["score"] for r in jl(E / f"e2e_judge_{j}_R2.jsonl") if r["score"] is not None} for j in J}
keys = [k for k in gen if all(k in sc[j] for j in J)]
panel = {k: sum(sc[j][k] >= 4 for j in J) >= 2 for k in keys}
bad = [k for k in keys if not panel[k]]
good = [k for k in keys if panel[k]]
random.Random(3).shuffle(good)
rows = bad + good[:60 - len(bad)]
print("rows", len(rows), "panel-wrong", len(bad), flush=True)

model = sys.argv[1]
extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
out = HERE / f"work/cand_{model}.jsonl"
done = {r["key"]: r for r in jl(out)}
for k in rows:
    if k in done and done[k]["score"] is not None:
        continue
    p = prompts.ANSWER_CORRECTNESS.format(reference=ctx[k]["reference"], answer=gen[k]["answer"])
    score, secs, err = None, None, None
    for attempt in range(6):
        t0 = time.time()
        r = requests.post("https://api.mistral.ai/v1/chat/completions", headers={"Authorization": "Bearer " + os.environ["MISTRAL_API_KEY"]}, timeout=120,
                          json={"model": model, "temperature": 0, "max_tokens": 1500, "messages": [{"role": "user", "content": p}], **extra})
        secs = round(time.time() - t0, 1)
        if r.status_code == 200:
            c = r.json()["choices"][0]["message"]["content"]
            if isinstance(c, list):
                c = "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
            m = re.search(r"\{.*\}", c or "", re.S)
            try:
                v = json.loads(m.group(0))["score"]
                score = int(v) if v == int(v) and 1 <= v <= 5 else None
            except (ValueError, KeyError, TypeError, AttributeError):
                err = "parse"
            break
        err = r.status_code
        time.sleep(20 if r.status_code == 429 else 5)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": k, "score": score, "secs": secs, "err": err}) + "\n")
    time.sleep(1.1)

res = {r["key"]: r for r in jl(out)}
ok = [k for k in rows if res.get(k, {}).get("score") is not None]
cg = {k: res[k]["score"] >= 4 for k in ok}
agree_panel = sum(cg[k] == panel[k] for k in ok)
agree_14b = sum(cg[k] == (sc["mistral"][k] >= 4) for k in ok)
exact_14b = sum(res[k]["score"] == sc["mistral"][k] for k in ok)
tp = sum(cg[k] and panel[k] for k in ok); fp = sum(cg[k] and not panel[k] for k in ok); fn = sum((not cg[k]) and panel[k] for k in ok); tn = sum((not cg[k]) and not panel[k] for k in ok)
n = len(ok)
po = (tp + tn) / n; pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
secs = sorted(res[k]["secs"] for k in ok if res[k]["secs"])
print(f"{model}: graded {n} of {len(rows)} | same good/not-good as panel {agree_panel}/{n} | as ministral-14b {agree_14b}/{n} | exact same 1-5 score as ministral-14b {exact_14b}/{n} | "
      f"vs panel: panel-wrong answers it calls good (missed) {fp} of {fp + tn}, panel-good it calls wrong {fn} of {tp + fn} | kappa {(po - pe) / (1 - pe) if pe < 1 else 1:.2f} | median {secs[len(secs) // 2]}s")
