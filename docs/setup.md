# Setup

How to run TriagePilot on your own machine. The first three steps build the search index and are tested end to end. Steps 4 and 5 (importing the workflows into n8n and sending a request) follow the exported workflows but were not repeated on a fresh n8n installation for this guide, so check them against what you see on screen.

You need Docker, Python 3.11 or newer, and one NVIDIA API key from build.nvidia.com. A Telegram bot is optional and only used for hand-off alerts.

## 1. Start the services

```bash
cd infra
cp .env.example .env        # set N8N_ENCRYPTION_KEY to a long random string
docker compose up -d
```

n8n runs on port 5678, Qdrant on 6333 and Jaeger (traces) on 16686. The ports are published on 127.0.0.1 only, so the services can be reached from this machine and not from the network. Change the mappings in `docker-compose.yml` to expose them. The folder `infra/local_files/` is mounted into n8n as `/home/node/.n8n-files`. The Wix workflow reads whole articles from there.

Expected result: `docker compose ps` lists three running containers, and `curl http://localhost:5678/healthz` returns `{"status":"ok"}`.

## 2. Download the articles

```bash
cd ../eval
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install pandas pyarrow numpy scipy requests pytest
python ingest/build_collections.py prepare
```

This downloads two public datasets from Hugging Face and writes one text file per article: 1,200 Wix help-center articles (WixQA, MIT licence) to `infra/local_files/wix-docs/` and 496 IBM technical notes (TechQA-RAG-Eval, Apache 2.0) to `infra/local_files/techqa-docs/`. Running it again produced files identical to the ones used for the reported results.

Expected result: 1,200 files in `infra/local_files/wix-docs/` and 496 in `infra/local_files/techqa-docs/`.

## 3. Build the search index

```bash
export NVIDIA_API_KEY=...          # on Windows PowerShell: $env:NVIDIA_API_KEY = "..."
python ingest/build_collections.py build wix
python ingest/build_collections.py build techqa
```

Each article is split into chunks of up to 1,000 characters with 200 overlap, the article title is added in front of each chunk, and every chunk is embedded with `nvidia/nemotron-3.5-embed-8b` (the first 2,048 values, normalised). The result is two Qdrant collections, `wix_help_center_docs_nemo` (5,964 chunks) and `techqa_docs_nemo` (3,772 chunks), which are the names the workflows search. A build takes about 15 minutes in total on a free NVIDIA key. The free tier limits requests per minute, so a slower or busier account takes longer. A build can be stopped and started again; finished chunks are kept in `eval/ingest/work/`.

Expected result: `curl http://localhost:6333/collections/wix_help_center_docs_nemo` reports 5,964 points, and the same call for `techqa_docs_nemo` reports 3,772.

To build into a second Qdrant without touching the first, add `--qdrant http://localhost:6340 --suffix _test`.

### What was checked

- The chunker is a port of the splitter n8n uses, because the Python LangChain splitter reproduces only about 80% of its chunks. Against the chunks of the running system, the port gives identical text and source for all 5,964 Wix chunks and all 3,772 TechQA chunks, and identical titles. Run `python ingest/build_collections.py check` to repeat this. It compares the chunks with the saved export of the live chunks in `eval/results/embedding_test/work/`.
- The rebuilt collections were compared with the running ones. For 100 random chunks per dataset, the closest chunk in the running collection is the same chunk, with cosine similarity 1.0000, and the top 6 results match in content and order (100 of 100 for both datasets).
- With 30 real questions per dataset, embedded the way the workflow does it: the top-1 chunk matched on 60 of 60 questions, the top 6 matched on 59 of 60. In the one that differed, the 6th and 7th chunks scored 0.54774 and 0.54774 against 0.54768, a difference in the fifth decimal that swapped them.
- The `line` field in each chunk's payload (the line of the article where the chunk starts) matches the running collection for about 93% of chunks. The workflows do not read it.

## 4. Import the workflows into n8n

Open http://localhost:5678, create the owner account, then import `workflows/triage-wix.json` and `workflows/triage-techqa.json` (Workflows, Import from file). Create these credentials and attach them to the nodes that use them:

| Credential type | Used by | Value |
|---|---|---|
| Header Auth | Webhook | Name `X-TriagePilot-Key`, a secret of your choice |
| Header Auth | Embed Question, Answer | Name `Authorization`, value `Bearer <your NVIDIA key>` |
| OpenAI API | Guardrails Chat Model | Base URL `https://integrate.api.nvidia.com/v1`, your NVIDIA key |
| Telegram API (optional) | Alert Telegram | Your bot token. Replace `YOUR_TELEGRAM_CHAT_ID` in the node with your chat ID. |

Then activate both workflows.

## 5. Send a request

```bash
curl -s -X POST http://localhost:5678/webhook/triage-techqa \
  -H "X-TriagePilot-Key: <your secret>" \
  -H "Content-Type: application/json" \
  -d '{"text": "How do I remove -Xcompressedrefs from WebSphere Application Server?"}'
```

The Wix workflow listens on `/webhook/triage-wix`. A request returns an answer, or a hand-off to a person when the system should not answer. The trace of each request is in Jaeger at http://localhost:16686.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| n8n or Qdrant does not answer | Docker may have stopped after a restart. Run `docker compose up -d` in `infra/` again. The Qdrant data is kept. |
| The webhook returns 403 | The `X-TriagePilot-Key` header is missing or does not match the secret in the Header Auth credential. |
| The index build stops with errors from the embedding endpoint | Run the same `build` command again. Finished chunks are kept, so it continues where it stopped. |
| A request takes minutes | The answer call waits on the model endpoint. A shared free endpoint can be slow, and a dedicated one removes most of the wait ([operations](operations.md)). |
