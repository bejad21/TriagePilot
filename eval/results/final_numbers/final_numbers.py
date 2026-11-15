"""Final headline numbers of the live run (200 Wix + 200 TechQA questions), from the saved answers and judge grades. Writes README.md next to this file.
usage (from eval/): python results/final_numbers/final_numbers.py
Headline = the current panel (nemotron, glm45, laguna), shown next to the same panel without the 8 unclear-label rows."""
import json
import pathlib
import random
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness import panel  # noqa: E402

W = HERE.parent / "live_best/work"
UNCLEAR = {  # label looks unfair for the question; kept in the headline, listed with reasons in unclear_labels.md
    "wix": {"wix:59", "wix:144", "wix:185", "wix:197"},
    "techqa": {"techqa:31", "techqa:36", "techqa:179", "techqa:216"},
}


def jl(p):
    p = pathlib.Path(p)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


live = {ds: {r["key"]: r for r in jl(W / f"live_{ds}.jsonl")} for ds in ("wix", "techqa")}
scores = {j: {r["key"]: r["score"] for r in jl(W / f"judge_{j}.jsonl") if r["score"] is not None} for j in ("nemotron", "glm45", "laguna")}


def ci(values, n=5000):
    rnd = random.Random(0)
    v = list(values)
    means = sorted(statistics.fmean(v[rnd.randrange(len(v))] for _ in v) for _ in range(n))
    return statistics.fmean(v), means[int(0.025 * n)], means[int(0.975 * n)]


def row(ds, judges, drop=frozenset()):
    keys = [k for k in live[ds] if k not in drop]
    answered = [k for k in keys if live[ds][k]["status"] == "answered" and live[ds][k]["answer"].strip()]
    good = {k: panel.is_good([scores[j].get(k) for j in judges]) for k in answered}
    gs = sum(good.values())
    ws = len(answered) - gs
    sc = ci([(1 if good[k] else -1) if k in good else 0 for k in keys])
    return len(keys), len(answered), len(keys) - len(answered), gs, ws, 100 * ws / len(answered), sc


out = ["# Final numbers of the live run", "",
       "Source: `../live_best/work/` (200 Wix and 200 TechQA questions sent through the live webhooks, answers and judge grades saved). "
       "Score = (good answers sent - wrong answers sent) / questions. Good = at least 2 of 3 judges give 4 or more. Run-to-run noise is about 0.04 to 0.05.", ""]
for title, judges, with_drop in (
        ("Headline: current panel (Nemotron-3.5-super, GLM-4.5-flash, Laguna-xs-2.1)", panel.PANEL, False),
        ("Same panel without the 8 unclear-label rows (4 Wix, 4 TechQA)", panel.PANEL, True)):
    out += [f"## {title}", "", "| Dataset | Questions | Answered | Handed off | Good sent | Wrong sent | Wrong share of sent | Score (95% interval) |", "|---|---|---|---|---|---|---|---|"]
    for ds, label in (("wix", "Wix"), ("techqa", "TechQA")):
        n, a, h, g, w, share, sc = row(ds, judges, UNCLEAR[ds] if with_drop else frozenset())
        out.append(f"| {label} | {n} | {a} | {h} | {g} | {w} | {share:.1f}% | {sc[0]:+.3f} ({sc[1]:+.3f} to {sc[2]:+.3f}) |")
    out.append("")
(HERE / "README.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
