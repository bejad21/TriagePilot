# TriagePilot — Shared Data Contracts

This file is the single source of truth for field names and shapes that must stay identical
across the Wix and TechQA workflows. When one workflow produces a field another workflow
consumes, the name and type are defined here once, so a rename in one place doesn't silently
break the other.

## Agent structured output (Root AI Agent and Resolution Sub-Agent)

Both agents must return exactly this shape:

```json
{
  "answer": "string — the answer text, or empty if needs_handoff/needs_human_review is true",
  "confidence": "high | medium | low",
  "needs_handoff": "boolean — root agent only; true if the resolution sub-agent should be called",
  "needs_human_review": "boolean — set by either agent; true if no automated answer should ship",
  "cited_source": "string — the doc filename/section the answer is grounded in, or empty if none"
}
```

Rules:
- `needs_handoff` is only ever set by the root agent. The resolution sub-agent never sets it,
  since there's no further agent to hand off to — it sets `needs_human_review` instead when it
  still can't answer confidently.
- `confidence: "low"` and `needs_human_review: true` both route to the escalation path, but they
  are not the same signal. Low confidence with a real attempted answer still gets shown to the
  human reviewer alongside the question; `needs_human_review: true` with an empty `answer` means
  the agent made no attempt.
- `cited_source` must be a real, checkable reference (a doc filename/section from the ingested
  corpus), never a fabricated-sounding citation. If the agent can't point to a real source,
  `confidence` should be "low", not a confident answer with no source.

## Webhook request shape

```json
{ "text": "string" }
```

## Webhook response shape

```json
{ "status": "answered", "answer": "string", "cited_source": "string" }
{ "status": "escalated", "message": "string" }
{ "status": "rejected", "reason": "string" }
```

`status` is the field every consumer (curl tests, the eval script) should branch on first.

## Evaluation dataset shape

```
eval/wix_eval_expertwritten.json:  question, expected_answer, source_url
eval/techqa_eval.json:             question, expected_answer, source_url
```

`expected_answer` is a real, expert-reviewed or ground-truth resolution from the source
benchmark (`Wix/WixQA` and `nvidia/TechQA-RAG-Eval` respectively), not a synthetic answer.
`source_url` links each question to its real source document — used only to build and score the
eval set, never fed to the retriever at inference time. `question` and `expected_answer` are the
two fields each Evaluation workflow compares the live system's `answer` output against.
