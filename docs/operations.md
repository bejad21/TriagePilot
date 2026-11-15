# Operations: evidence that it ran

This is a self-hosted deployment that was evaluated on benchmark questions, not on real customer traffic. This page shows what the running system did, how long it took, and what went wrong along the way.

## What ran

- **Live since 7 Nov 2026.** Both workflows run on local Docker Compose (n8n, Qdrant, Jaeger). The exports of the live workflows are in `workflows/`.
- **400 evaluation requests** went through the live webhooks on 8 Nov 2026 (200 Wix, 200 TechQA). All 400 ended with a valid response: 359 answered and 41 escalated to a person ([evaluation](evaluation.md)).
- **84 attack requests** (42 on each webhook) were sent to the live webhooks. All were rejected ([security tests](security-tests.md)).
- **Time per request:** median 161 s on Wix and 190 s on TechQA, almost all of it in the answer model call on a shared free endpoint.

## One request, traced

n8n exports an OpenTelemetry trace per request to Jaeger. This is a real TechQA request from 11 Nov 2026, with one span per node (durations in seconds):

| Start | Node | Takes |
|---|---|---|
| 0.0 | Webhook | 0.00 |
| 0.0 | Route by Modality, Normalize Text | 0.01 |
| 0.0 | Jailbreak Intent Check (guard model) | 0.43 |
| 0.5 | Redact PII | 0.04 |
| 0.5 | Embed Question | 0.71 |
| 1.2 | Search Qdrant | 0.09 |
| 1.3 | Build Context | 0.02 |
| 1.3 | **Answer (mistral-large-4)** | **41.13** |
| 42.5 | Parse Answer | 0.03 |
| 42.5 | Check Retrieval | 0.02 |
| 42.5 | Check Outbound Content (guard model) | 1.02 |
| 43.5 | Should Escalate | 0.02 |
| 43.6 | Respond Answered | 0.01 |
| | **Whole request** | **43.6** |

The trace comes from the running workflow. The files in `workflows/` are exports of the same nodes and connections, cleaned for publication (owner details removed, the Telegram chat id replaced by a placeholder, the parse step named "Parse Answer").

Another request the same day took 157.3 s in total, of which 150.4 s was the answer call. Everything except the answer call took 2.5 seconds in the first trace and 6.9 seconds in the second. That is why the latency numbers in the evaluation reflect the hosting of one model call.

## Incidents and what changed

| Date | What happened | What changed |
|---|---|---|
| 5 to 6 Nov 2026 | The hosted embedding endpoint returned errors for about 13 minutes after thousands of calls in a row. | The embedding and evaluation jobs are resumable, so a run continues where it stopped. |
| 7 to 8 Nov 2026 | The guard checks on the first provider ran at about one question a minute, which would have capped the whole system. | Both guard checks moved to `nemotron-3.5-bolt-e4b`. 60 models were screened, and the live webhooks rejected 42 of 42 attacks afterwards. |
| 7 to 8 Nov 2026 | The shared answer endpoint was slow: a one-word reply took about 138 s. | Request timeouts were set to 600 s in the live evaluation and 420 s in offline runs, with low parallelism. Live latency is reported as measured. |
| 10 Nov 2026 | Hybrid search lost the right article on 13 of 200 Wix questions. | Not adopted ([experiments](experiments.md)). |
| 11 Nov 2026 | The same TechQA question (row 52) cited a look-alike document on 8 Nov and the right one on a re-run; the search returns the right document in both cases. | Counted as run-to-run noise (see the noise estimate in the evaluation, about 0.04 to 0.05), not as a defect. |
| 11 Nov 2026 | Docker Desktop had stopped after a restart, so n8n and Qdrant were down. | The three containers were started again. n8n answered its health check after 16 seconds, and the TechQA collection still held all 3,772 vectors. Two live requests succeeded straight away. |

## Reproducing a run

```bash
cd infra
docker compose up -d
curl -s http://localhost:5678/healthz
curl -s http://localhost:6333/collections/techqa_docs_nemo

curl -s -X POST http://localhost:5678/webhook/triage-techqa \
  -H "X-TriagePilot-Key: $TRIAGEPILOT_WEBHOOK_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "How do I remove -Xcompressedrefs from WebSphere Application Server?"}'
```

The evaluation scripts that produced the 400 requests are in `eval/results/live_best/` (`live_eval.py`, with saved answers and judge grades in `work/`).

## What is missing for real traffic

- Alerting beyond the Telegram message on an escalation.
- A feedback loop, so that corrected answers improve the next ones.
- Load testing against a dedicated model endpoint, since the free shared endpoint used here sets the response time.
