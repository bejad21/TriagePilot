"""Send the 42 labelled attacks of the earlier guardrail tests to a live webhook and count how many are rejected (expected: all).
usage (from eval/, TRIAGEPILOT_WEBHOOK_KEY loaded): python results/live_best/guard_probe.py triage-wix
Look-alikes (normal questions that sound like attacks) are not sent here: each one runs the full answer pipeline; guard_offline.py runs them against the guard model directly (1 of 188 normal inputs flagged).
"""
import concurrent.futures as cf
import json
import os
import pathlib
import sys
from collections import Counter

import requests

HERE = pathlib.Path(__file__).resolve().parent

path = sys.argv[1]
attacks = [(i["id"], i["text"]) for i in json.loads((HERE / "guard_inputs.json").read_text(encoding="utf-8")) if i["label"] == "attack"]


def send(kt):
    k, t = kt
    try:
        r = requests.post(f"http://localhost:5678/webhook/{path}", json={"text": t}, headers={"X-TriagePilot-Key": os.environ["TRIAGEPILOT_WEBHOOK_KEY"]}, timeout=600)
        return k, r.json().get("status", f"http_{r.status_code}") if r.text.strip() else f"empty_{r.status_code}", t
    except (requests.RequestException, ValueError) as e:
        return k, type(e).__name__, t


with cf.ThreadPoolExecutor(3) as ex:
    out = list(ex.map(send, attacks))
print(path, "attacks sent", len(out), dict(Counter(s for _, s, _ in out)))
for k, s, t in out:
    if s != "rejected":
        print("  not rejected:", k, s, "|", t[:110].replace("\n", " "))
