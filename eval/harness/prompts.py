"""Judge prompts ported verbatim from workflows/08-evaluation-wix.json.

Source nodes: "Answer Correctness Judge", "Generate Reverse Questions",
"Faithfulness Judge", "Context Precision Judge", "Context Recall Judge".
n8n expressions became str.format fields and literal braces are doubled.
The only wording change is the first line of ANSWER_CORRECTNESS: Wix and
TechQA (workflows/11-evaluation-techqa.json) differ only in the data set named
there, so it is neutral here.
"""

ANSWER_CORRECTNESS = (
    "Ground truth (expert-written answer for a real customer support question):\n"
    "{reference}\n\n"
    "System's actual answer:\n"
    "{answer}\n\n"
    "Rate how consistent the actual answer is with this specific ground truth "
    "resolution on a scale of 1 (contradicts or unrelated) to 5 (fully consistent, "
    "no unsupported claims). Note: the ground truth is one real person's resolution "
    "and may reference environment-specific details (a config value, a PR, an "
    "external link) the system cannot be expected to reproduce verbatim - score "
    "based on substantive consistency with the underlying fix, not exact wording. "
    'Return ONLY JSON: {{"score": <1-5 integer>}}.'
)

REVERSE_QUESTIONS = (
    "You are evaluating answer relevancy using the RAGAS methodology: given an "
    "answer, generate questions that this answer would be a good, complete response "
    "to (reverse-engineering the question from the answer).\n\n"
    "System's actual answer:\n"
    "{answer}\n\n"
    "Generate exactly 3 different questions that this answer directly and "
    "completely responds to. If the answer is noncommittal, vague, or a "
    "refusal/escalation with no real informational content (e.g. \"this has been "
    "routed to a human for review\"), return an empty list instead - a noncommittal "
    "answer cannot be reverse-engineered into a real question.\n\n"
    'Return ONLY JSON: {{"questions": ["<question 1>", "<question 2>", '
    '"<question 3>"]}} or {{"questions": []}} for noncommittal answers.'
)

FAITHFULNESS = (
    "You are evaluating faithfulness using the RAGAS methodology: decompose the "
    "system's answer into distinct atomic factual claims, then judge whether EACH "
    "claim is directly supported by the retrieved context. This is claim-level "
    "grounding checking, not a holistic 1-5 rating.\n\n"
    "System's actual answer:\n"
    "{answer}\n\n"
    "Retrieved documentation context (what the system's RAG retriever actually "
    "found for this question):\n"
    "{context}\n\n"
    "Steps:\n"
    "1. Break the answer into a list of distinct, atomic factual claims (a claim "
    "that combines two facts should be split into two claims).\n"
    "2. For each claim, judge: is it directly supported/entailed by the retrieved "
    "context above (true), or not supported / contradicted / absent from the "
    "context (false)?\n\n"
    "This is independent of whether the answer is a good answer to the question or "
    "matches any external ground truth - only check whether each claim is grounded "
    "in the retrieved context provided above.\n\n"
    'Return ONLY JSON: {{"claims": [{{"claim": "<claim text>", "supported": '
    'true|false}}, ...]}}. If the answer makes no factual claims (e.g. it\'s a '
    'pure refusal), return {{"claims": []}}.'
)

CONTEXT_PRECISION = (
    "You are evaluating context precision using the RAGAS methodology: for each "
    "retrieved chunk below (numbered, in retrieval rank order), judge whether it is "
    "relevant to answering the question - i.e. whether it would help produce the "
    "reference answer.\n\n"
    "Question:\n"
    "{question}\n\n"
    "Reference answer (ground truth):\n"
    "{reference}\n\n"
    "Retrieved chunks, in rank order (chunk 1 is the top-ranked result):\n"
    "{chunks}\n\n"
    'Return ONLY JSON: {{"relevance": [true|false, true|false, ...]}} - exactly one '
    "boolean per chunk, in the same order as listed above."
)

CONTEXT_RECALL = (
    "You are evaluating context recall using the RAGAS methodology: decompose the "
    "REFERENCE answer (ground truth) into distinct atomic factual claims, then "
    "judge whether each claim can be attributed to (is supported by) the retrieved "
    "context.\n\n"
    "Reference answer (ground truth):\n"
    "{reference}\n\n"
    "Retrieved context:\n"
    "{context}\n\n"
    "Steps:\n"
    "1. Break the reference answer into a list of distinct, atomic factual claims.\n"
    "2. For each claim, judge: is it attributable to / supported by the retrieved "
    "context above (true), or not present in the retrieved context (false)?\n\n"
    'Return ONLY JSON: {{"claims": [{{"claim": "<claim text>", "attributed": '
    'true|false}}, ...]}}.'
)
