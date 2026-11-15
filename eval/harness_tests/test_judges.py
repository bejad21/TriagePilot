import json
import math

import pytest

from harness import judges, prompts
from harness.judges import JudgeParseError


class FakeChat:
    name = "fake"

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[tuple[str, str, float]] = []

    def complete(self, system, user, *, temperature=0.0):
        self.calls.append((system, user, temperature))
        return self.replies.pop(0)


class FakeEmbedder:
    name = "fake-embed"

    def __init__(self, vectors):
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return self.vectors


def claims_json(key, flags):
    return json.dumps({"claims": [{"claim": f"c{i}", key: f} for i, f in enumerate(flags)]})


def test_parse_json_object_plain_fenced_and_prose():
    assert judges.parse_json_object('{"a": 1}') == {"a": 1}
    assert judges.parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert judges.parse_json_object('Sure! {"a": {"b": 2}} Hope that helps.') == {"a": {"b": 2}}


def test_parse_json_object_closes_a_reply_missing_its_last_brace():
    text = '{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}]'
    assert judges.parse_json_object(text) == {"claims": [
        {"claim": "a", "supported": True}, {"claim": "b", "supported": False}]}


@pytest.mark.parametrize("text", ["", "not json", "[1, 2]", "{broken", '"str"'])
def test_parse_json_object_raises(text):
    with pytest.raises(JudgeParseError):
        judges.parse_json_object(text)


def test_faithfulness_three_of_four():
    chat = FakeChat(claims_json("supported", [True, True, False, True]))
    assert judges.score_faithfulness(chat, "ans", ["ctx a", "ctx b"]) == 0.75
    assert len(chat.calls) == 1
    user = chat.calls[0][1]
    assert "ans" in user and "ctx a\n\n---\n\nctx b" in user


def test_faithfulness_zero_claims_is_none():
    assert judges.score_faithfulness(FakeChat('{"claims": []}'), "a", ["c"]) is None


@pytest.mark.parametrize("reply", [
    "garbage",
    '{"other": []}',
    '{"claims": "x"}',
    '{"claims": [{"claim": "c"}]}',
    '{"claims": [{"claim": "c", "supported": "yes"}]}',
])
def test_faithfulness_bad_output_raises(reply):
    with pytest.raises(JudgeParseError):
        judges.score_faithfulness(FakeChat(reply), "a", ["c"])


def test_context_precision_average_precision():
    # relevance [T, F, T]: precision@1 = 1/1, precision@3 = 2/3 over relevant
    # ranks only; mean = (1 + 2/3) / 2 = 5/6.
    chat = FakeChat('{"relevance": [true, false, true]}')
    score = judges.score_context_precision(chat, "q", "ref", ["a", "b", "c"])
    assert math.isclose(score, 5 / 6)
    assert len(chat.calls) == 1
    user = chat.calls[0][1]
    assert "Chunk 1:\na\n\n---\n\nChunk 2:\nb" in user
    assert "q" in user and "ref" in user


def test_context_precision_none_relevant_is_zero():
    chat = FakeChat('{"relevance": [false, false]}')
    assert judges.score_context_precision(chat, "q", "r", ["a", "b"]) == 0.0


def test_context_precision_empty_contexts_no_call():
    chat = FakeChat()
    assert judges.score_context_precision(chat, "q", "r", []) == 0.0
    assert chat.calls == []


@pytest.mark.parametrize("reply", [
    '{"relevance": [true]}',
    '{"relevance": [true, "no"]}',
    '{"relevance": "x"}',
    "nope",
])
def test_context_precision_bad_output_raises(reply):
    with pytest.raises(JudgeParseError):
        judges.score_context_precision(FakeChat(reply), "q", "r", ["a", "b"])


def test_context_recall_two_of_five():
    chat = FakeChat(claims_json("attributed", [True, False, True, False, False]))
    assert judges.score_context_recall(chat, "ref", ["c"]) == 0.4
    assert len(chat.calls) == 1


def test_context_recall_zero_claims_is_none():
    assert judges.score_context_recall(FakeChat('{"claims": []}'), "r", ["c"]) is None


def test_context_recall_wrong_key_raises():
    with pytest.raises(JudgeParseError):
        judges.score_context_recall(FakeChat(claims_json("supported", [True])), "r", ["c"])


@pytest.mark.parametrize("n", [1, 3, 5])
def test_answer_correctness_valid(n):
    chat = FakeChat(json.dumps({"score": n}))
    assert judges.score_answer_correctness(chat, "q", "ref", "ans") == n
    assert len(chat.calls) == 1
    assert "ref" in chat.calls[0][1] and "ans" in chat.calls[0][1]


def test_answer_correctness_accepts_integral_float():
    assert judges.score_answer_correctness(FakeChat('{"score": 4.0}'), "q", "r", "a") == 4


@pytest.mark.parametrize("reply", [
    '{"score": 0}', '{"score": 6}', '{"score": 3.5}', '{"score": "4"}',
    '{"score": true}', '{"x": 3}', "nonsense",
    '{"score": NaN}', '{"score": Infinity}', '{"score": -Infinity}',
])
def test_answer_correctness_invalid_raises(reply):
    with pytest.raises(JudgeParseError):
        judges.score_answer_correctness(FakeChat(reply), "q", "r", "a")


def test_answer_relevancy_mean_cosine():
    chat = FakeChat('{"questions": ["r1", "r2"]}')
    # cos([1,0],[1,0]) = 1; cos([1,0],[1,1]) = 1/sqrt(2); mean = (1 + 0.70710678) / 2
    embedder = FakeEmbedder([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    score = judges.score_answer_relevancy(chat, embedder, "q", "ans")
    assert math.isclose(score, (1 + 1 / math.sqrt(2)) / 2)
    assert embedder.calls == [["q", "r1", "r2"]]
    assert len(chat.calls) == 1


def test_answer_relevancy_caps_at_three_questions():
    chat = FakeChat('{"questions": ["a", "b", "c", "d"]}')
    embedder = FakeEmbedder([[1.0]] * 4)
    judges.score_answer_relevancy(chat, embedder, "q", "ans")
    assert embedder.calls == [["q", "a", "b", "c"]]


def test_answer_relevancy_no_questions_is_none_without_embedding():
    embedder = FakeEmbedder([])
    assert judges.score_answer_relevancy(FakeChat('{"questions": []}'), embedder, "q", "a") is None
    assert embedder.calls == []


def test_answer_relevancy_short_embedding_result_is_none():
    chat = FakeChat('{"questions": ["r1"]}')
    assert judges.score_answer_relevancy(chat, FakeEmbedder([[1.0, 0.0]]), "q", "a") is None


def test_answer_relevancy_zero_vector_scores_zero_similarity():
    chat = FakeChat('{"questions": ["r1"]}')
    embedder = FakeEmbedder([[0.0, 0.0], [1.0, 0.0]])
    assert judges.score_answer_relevancy(chat, embedder, "q", "a") == 0.0


@pytest.mark.parametrize("reply", ['{"questions": "x"}', '{"questions": [1]}', "nope"])
def test_answer_relevancy_bad_output_raises(reply):
    with pytest.raises(JudgeParseError):
        judges.score_answer_relevancy(FakeChat(reply), FakeEmbedder([]), "q", "a")


def test_prompts_keep_json_instructions():
    assert '{"score": <1-5 integer>}' in prompts.ANSWER_CORRECTNESS.format(
        reference="r", answer="a")
