"""Question side of the embedding test: the 300 questions, a mistral-large-4 rewrite and hypothetical answer for each, and the query vectors.

usage (from eval/, keys loaded):
  python results/embedding_test/queries.py gen            mistral-large-4 (NVIDIA, reasoning low, 10 at once): search query + hypothetical help-article passage per question; resumable
  python results/embedding_test/queries.py mistral        mistral-embed vectors of the customer's questions (baseline); resumable
  python results/embedding_test/queries.py nemo           Nemotron 8B vectors of: question (query), rewrite (query), hypothetical answer (passage and query type)
Questions: Wix 200 (ids_wix.txt + ids_wix_fresh.txt), TechQA 100 (ids_techqa.txt); keys "wix:<id>", "techqa:<id>".
The prompt names no product, feature or test question; it only asks for terminology and the shape of an article.
"""
import concurrent.futures as cf
import json
import os
import pathlib
import re
import sys
import time

import requests

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.nvidia_key import api_key as nvidia_key  # noqa: E402
RES = HERE.parent
EVAL = RES.parent
WORK = HERE / "work"
IDS = {"wix": (RES / "final_test/work/ids_wix.txt", RES / "fullarticle_copy/work/ids_wix_fresh.txt"), "techqa": (RES / "final_test/work/ids_techqa.txt",)}
DOMAIN = {"wix": "the Wix website builder help center", "techqa": "IBM technical support Technotes"}
SYSTEM = """You help a search engine find the right article in {domain}. You get a customer's question. Return only a JSON object with two keys:
"search_query": a short standalone search query (4 to 14 words) that uses the terminology a help article would use for the feature, product or error involved; no filler words, no greeting.
"hypothetical_answer": 3 to 5 sentences written the way the help article that answers this question would be written (headings and steps style). Do not invent version numbers, prices, links or exact menu names you are unsure of."""


def questions():
    out = {}
    for ds, files in IDS.items():
        data = {str(r["id"]): r for r in json.loads((EVAL / f"data/{ds}_full.json").read_text(encoding="utf-8"))}
        for f in files:
            for rid in f.read_text().split(","):
                gold = data[rid]["gold_sources"]
                out[f"{ds}:{rid}"] = {"ds": ds, "question": data[rid]["question"], "gold": gold if isinstance(gold, list) else eval(gold)}
    return out


def jl(p):
    p = pathlib.Path(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def ml4(key, q):
    body = {"model": "mistralai/mistral-large-4-1t-a52b", "temperature": 0, "max_tokens": 1200, "reasoning_effort": "low",
            "messages": [{"role": "system", "content": SYSTEM.format(domain=DOMAIN[q["ds"]])}, {"role": "user", "content": q["question"][:3000]}]}
    for attempt in range(5):
        try:
            r = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers={"Authorization": "Bearer " + nvidia_key()}, json=body, timeout=150)
            if r.status_code == 200:
                raw = r.json()["choices"][0]["message"].get("content") or ""
                m = re.search(r"\{.*\}", raw, re.S)
                try:
                    d = json.loads(m.group(0)) if m else {}
                except ValueError:
                    d = {}
                if d.get("search_query") and d.get("hypothetical_answer"):
                    return {"key": key, "search_query": str(d["search_query"]), "hyde": str(d["hypothetical_answer"])}
        except requests.RequestException:
            pass
        time.sleep(6 * (attempt + 1))
    return {"key": key, "search_query": "", "hyde": ""}


def gen():
    qs = questions()
    out = WORK / "queries_ml4.jsonl"
    rows = jl(out)
    done = {r["key"] for r in rows if r["search_query"]}
    out.write_text("".join(json.dumps(r) + "\n" for r in rows if r["search_query"]), encoding="utf-8")
    todo = [(k, q) for k, q in qs.items() if k not in done]
    print("questions", len(qs), "to generate", len(todo), flush=True)
    with cf.ThreadPoolExecutor(int(os.environ.get("GEN_WORKERS", "10"))) as ex:
        for r in ex.map(lambda kq: ml4(*kq), todo):
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("gen done; failed:", [r["key"] for r in jl(out) if not r["search_query"]], flush=True)


def mistral():
    qs = questions()
    cache_p = WORK / "q_mistral.json"
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
    for k, q in qs.items():
        if k in cache:
            continue
        for attempt in range(8):
            time.sleep(2.2)
            r = requests.post("https://api.mistral.ai/v1/embeddings", headers={"Authorization": "Bearer " + os.environ["MISTRAL_API_KEY"]}, json={"model": "mistral-embed", "input": [q["question"][:6000]]}, timeout=60)
            if r.status_code == 200:
                cache[k] = r.json()["data"][0]["embedding"]
                break
            time.sleep(10 * (attempt + 1))
        if len(cache) % 25 == 0:
            cache_p.write_text(json.dumps(cache), encoding="utf-8")
    cache_p.write_text(json.dumps(cache), encoding="utf-8")
    print("mistral question vectors", len(cache), "of", len(qs))


def nemo_vec(text, kind):
    body = {"model": "nvidia/nemotron-3.5-embed-8b", "input": [text[:6000]], "input_type": kind, "truncate": "END"}
    for attempt in range(8):
        try:
            r = requests.post("https://integrate.api.nvidia.com/v1/embeddings", headers={"Authorization": "Bearer " + nvidia_key()}, json=body, timeout=120)
            if r.status_code == 200:
                return r.json()["data"][0]["embedding"]
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return None


def nemo():
    qs = questions()
    ml = {r["key"]: r for r in jl(WORK / "queries_ml4.jsonl")}
    jobs = []
    for k, q in qs.items():
        jobs.append((k, "question", q["question"], "query"))
        if k in ml and ml[k]["search_query"]:
            jobs += [(k, "rewrite", ml[k]["search_query"], "query"), (k, "hyde_passage", ml[k]["hyde"], "passage"), (k, "hyde_query", ml[k]["hyde"], "query")]
    out_p = WORK / "q_nemo.json"
    cache = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    todo = [j for j in jobs if f"{j[0]}|{j[1]}" not in cache]
    with cf.ThreadPoolExecutor(12) as ex:
        for (k, kind, _t, _i), v in zip(todo, ex.map(lambda j: nemo_vec(j[2], j[3]), todo)):
            if v is not None:
                cache[f"{k}|{kind}"] = v
    out_p.write_text(json.dumps(cache), encoding="utf-8")
    print("nemo query vectors", len(cache), "of", len(jobs))


if __name__ == "__main__":
    {"gen": gen, "mistral": mistral, "nemo": nemo}[sys.argv[1]]()
