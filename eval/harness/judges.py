"""LLM-judge scoring: faithfulness, context precision, context recall and answer relevancy (the RAGAS definitions).

- Unparsable or missing judge output raises JudgeParseError.
- score_answer_relevancy returns None when fewer than 2 vectors come back or the
  answer yields no reverse questions.
- score_context_precision raises when the relevance list does not have one
  entry per chunk.

Each scoring function makes one backend call per prompt; retries belong to the
backend. Prompts are sent as the user message with an empty system message.
"""
import json
import math
import re
from typing import Protocol

from harness import prompts

_SEPARATOR = "\n\n---\n\n"
_MAX_REVERSE_QUESTIONS = 3
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class ChatBackend(Protocol):
    name: str

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        """Return the raw model text for one prompt."""


class EmbeddingBackend(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in order."""


class JudgeParseError(ValueError):
    """Judge output was missing, unparsable, or outside its schema."""


def parse_json_object(text: str) -> dict:
    """Parse the first JSON object in text, tolerating code fences and prose."""
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE.findall(text)]
    decoder = json.JSONDecoder()
    # Tried first: a truncated reply would otherwise yield one of its inner objects.
    repaired = _close_truncated(text)
    if repaired is not None:
        return repaired
    for candidate in candidates:
        for start in (i for i, ch in enumerate(candidate) if ch == "{"):
            try:
                value, _ = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise JudgeParseError(f"no JSON object in judge output: {text[:200]!r}")


def _close_truncated(text: str) -> dict | None:
    """Parse a reply cut off before its final closing brackets, as some judges produce.

    Only a missing "}" or "]}" at the very end is added; nothing is invented.
    """
    start = text.find("{")
    if start < 0:
        return None
    body = text[start:].strip()
    for tail in ("}", "]}"):
        try:
            value = json.loads(body + tail)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _ask(backend: ChatBackend, prompt: str) -> dict:
    return parse_json_object(backend.complete("", prompt))


def _claim_flags(data: dict, key: str) -> list[bool]:
    claims = data.get("claims")
    if not isinstance(claims, list):
        raise JudgeParseError(f"'claims' must be a list, got {claims!r}")
    flags = []
    for item in claims:
        if not isinstance(item, dict) or not isinstance(item.get(key), bool):
            raise JudgeParseError(f"claim lacks boolean '{key}': {item!r}")
        flags.append(item[key])
    return flags


def score_faithfulness(backend: ChatBackend, answer: str,
                       contexts: list[str]) -> float | None:
    """Supported claims / total claims; None when the answer has no claims."""
    prompt = prompts.FAITHFULNESS.format(answer=answer, context=_SEPARATOR.join(contexts))
    flags = _claim_flags(_ask(backend, prompt), "supported")
    return sum(flags) / len(flags) if flags else None


def score_context_precision(backend: ChatBackend, question: str, reference: str,
                            contexts: list[str]) -> float:
    """Average precision@k over relevant chunks in rank order; 0.0 if none."""
    if not contexts:
        return 0.0
    chunks = _SEPARATOR.join(f"Chunk {i}:\n{c}" for i, c in enumerate(contexts, start=1))
    prompt = prompts.CONTEXT_PRECISION.format(
        question=question, reference=reference, chunks=chunks)
    relevance = _ask(backend, prompt).get("relevance")
    if (not isinstance(relevance, list) or len(relevance) != len(contexts)
            or not all(isinstance(r, bool) for r in relevance)):
        raise JudgeParseError(
            f"'relevance' must be {len(contexts)} booleans, got {relevance!r}")
    relevant = 0
    precision_sum = 0.0
    for k, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            relevant += 1
            precision_sum += relevant / k
    return precision_sum / relevant if relevant else 0.0


def score_context_recall(backend: ChatBackend, reference: str,
                         contexts: list[str]) -> float | None:
    """Attributed reference claims / total; None when there are no claims."""
    prompt = prompts.CONTEXT_RECALL.format(
        reference=reference, context=_SEPARATOR.join(contexts))
    flags = _claim_flags(_ask(backend, prompt), "attributed")
    return sum(flags) / len(flags) if flags else None


def score_answer_correctness(backend: ChatBackend, question: str, reference: str,
                             answer: str) -> int:
    """Integer 1 to 5 consistency with the reference answer.

    `question` is accepted for call-site symmetry; it is not part of the
    ported prompt.
    """
    prompt = prompts.ANSWER_CORRECTNESS.format(reference=reference, answer=answer)
    score = _ask(backend, prompt).get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise JudgeParseError(f"'score' must be a number, got {score!r}")
    if not math.isfinite(score) or score != int(score) or not 1 <= score <= 5:
        raise JudgeParseError(f"'score' must be an integer from 1 to 5, got {score!r}")
    return int(score)


def _cosine(a: list[float], b: list[float]) -> float:
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / denom if denom > 0 else 0.0


def score_answer_relevancy(chat: ChatBackend, embedder: EmbeddingBackend,
                           question: str, answer: str) -> float | None:
    """Mean cosine of the question vs up to 3 reverse questions; None if unavailable."""
    data = _ask(chat, prompts.REVERSE_QUESTIONS.format(answer=answer))
    questions = data.get("questions")
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        raise JudgeParseError(f"'questions' must be a list of strings, got {questions!r}")
    questions = questions[:_MAX_REVERSE_QUESTIONS]
    if not questions:
        return None
    vectors = embedder.embed([question, *questions])
    if len(vectors) < 2:
        return None
    sims = [_cosine(vectors[0], v) for v in vectors[1:]]
    return sum(sims) / len(sims)
