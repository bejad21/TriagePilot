import json

import pytest
import requests

from harness.backends import (BackendError, CachedChat, MistralChat, MistralEmbed,
                              OpenAICompatChat)

KEY = "test-key"


class FakeResponse:
    def __init__(self, status: int = 200, body=None, text: str | None = None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def chat_ok(content="hi"):
    return FakeResponse(200, {"choices": [{"message": {"content": content}}]})


class FakeSession:
    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[dict] = []

    def post(self, url, json, headers, timeout):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def key_env(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", KEY)


def make(session, **kw):
    sleeps: list[float] = []
    backend = OpenAICompatChat("m1", "FAKE_KEY", "https://api.example.com/v1/",
                               session=session, sleep=sleeps.append, **kw)
    return backend, sleeps


def test_success_returns_content_and_request_shape():
    session = FakeSession(chat_ok("hello"))
    backend, sleeps = make(session)
    assert backend.complete("sys", "usr", temperature=0.3) == "hello"
    call = session.calls[0]
    assert call["url"] == "https://api.example.com/v1/chat/completions"
    assert call["json"] == {"model": "m1", "temperature": 0.3, "messages": [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]}
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"
    assert sleeps == []


def test_empty_system_is_omitted():
    session = FakeSession(chat_ok())
    backend, _ = make(session)
    backend.complete("", "usr")
    assert session.calls[0]["json"]["messages"] == [{"role": "user", "content": "usr"}]


def test_name_strings():
    assert OpenAICompatChat("m1", "FAKE_KEY", "https://api.example.com/v1").name == \
        "api.example.com/m1"
    assert MistralChat().name == "api.mistral.ai/mistral-small-latest"
    assert MistralEmbed().name == "api.mistral.ai/mistral-embed"


def test_429_then_200_retries_once_with_sleep():
    session = FakeSession(FakeResponse(429, text="slow down"), chat_ok("ok"))
    backend, sleeps = make(session)
    assert backend.complete("", "u") == "ok"
    assert len(session.calls) == 2
    assert len(sleeps) == 1 and sleeps[0] > 0


def test_backoff_grows_exponentially():
    session = FakeSession(*[FakeResponse(503, text="x")] * 3, chat_ok())
    backend, sleeps = make(session)
    backend.complete("", "u")
    assert sleeps[0] < sleeps[1] < sleeps[2]


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retryable_statuses(status):
    session = FakeSession(FakeResponse(status, text="x"), chat_ok())
    backend, _ = make(session)
    assert backend.complete("", "u") == "hi"


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_non_retryable_statuses_fail_immediately(status):
    session = FakeSession(FakeResponse(status, text="nope"))
    backend, sleeps = make(session)
    with pytest.raises(BackendError, match=str(status)):
        backend.complete("", "u")
    assert len(session.calls) == 1 and sleeps == []


def test_connection_and_timeout_errors_are_retried():
    session = FakeSession(requests.ConnectionError("boom"), requests.Timeout("slow"),
                          chat_ok("ok"))
    backend, sleeps = make(session)
    assert backend.complete("", "u") == "ok"
    assert len(sleeps) == 2


def test_exhausted_retries_raise_with_status_and_truncated_body():
    session = FakeSession(*[FakeResponse(500, text="e" * 5000)] * 3)
    backend, sleeps = make(session, max_retries=2)
    with pytest.raises(BackendError) as err:
        backend.complete("", "u")
    assert len(session.calls) == 3 and len(sleeps) == 2
    assert "500" in str(err.value) and len(str(err.value)) < 1000


def test_exhausted_connection_errors_raise_backend_error():
    session = FakeSession(*[requests.ConnectionError("boom")] * 2)
    backend, _ = make(session, max_retries=1)
    with pytest.raises(BackendError):
        backend.complete("", "u")


def test_missing_env_var_raises_before_request(monkeypatch):
    monkeypatch.delenv("FAKE_KEY")
    session = FakeSession()
    backend, _ = make(session)
    with pytest.raises(BackendError, match="missing key env var FAKE_KEY"):
        backend.complete("", "u")
    assert session.calls == []


@pytest.mark.parametrize("response", [
    FakeResponse(200, {"choices": []}),
    FakeResponse(200, {"choices": [{"message": {"content": ""}}]}),
    FakeResponse(200, {"choices": [{"message": {"content": None}}]}),
    FakeResponse(200, {"error": "x"}),
    FakeResponse(200, text="<html>"),
])
def test_unusable_response_raises(response):
    backend, _ = make(FakeSession(response))
    with pytest.raises(BackendError):
        backend.complete("", "u")


def test_key_never_appears_in_errors():
    backend, _ = make(FakeSession(FakeResponse(401, text=f"bad key {KEY}")))
    with pytest.raises(BackendError) as err:
        backend.complete("", "u")
    assert KEY not in str(err.value) and "Bearer" not in str(err.value)
    backend, _ = make(FakeSession(requests.ConnectionError(f"Bearer {KEY}")))
    backend.max_retries = 0
    with pytest.raises(BackendError) as err:
        backend.complete("", "u")
    assert KEY not in str(err.value) and KEY not in repr(err.value.__cause__)


def test_key_redacted_when_retries_exhausted_on_echoing_body():
    echo = FakeResponse(500, text=f"Authorization: Bearer {KEY}")
    backend, _ = make(FakeSession(*[echo] * 3), max_retries=2)
    with pytest.raises(BackendError) as err:
        backend.complete("", "u")
    assert KEY not in str(err.value)


@pytest.mark.parametrize("response", [
    FakeResponse(200, {"error": f"invalid key {KEY}"}),
    FakeResponse(200, {"choices": [{"oops": f"Bearer {KEY}"}]}),
    FakeResponse(200, text=f"<html>{KEY}</html>"),
    FakeResponse(200, [KEY], text=f'["{KEY}"]'),
])
def test_key_redacted_from_malformed_success_body(response):
    backend, _ = make(FakeSession(response))
    with pytest.raises(BackendError) as err:
        backend.complete("", "u")
    assert KEY not in str(err.value)


def test_key_redacted_from_embedding_error(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", KEY)
    bad = FakeResponse(200, {"data": [{"index": 0, "embedding": [1.0], "note": KEY}] * 3})
    with pytest.raises(BackendError) as err:
        make_embed(FakeSession(bad)).embed(["a"])
    assert KEY not in str(err.value)


def embed_ok(n, offset=0):
    return FakeResponse(200, {"data": [{"index": i, "embedding": [float(offset + i)]}
                                       for i in range(n)]})


def make_embed(session):
    return MistralEmbed(session=session, sleep=lambda s: None)


def test_embed_batches_at_32(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", KEY)
    session = FakeSession(embed_ok(32), embed_ok(32, 100), embed_ok(1, 200))
    vectors = make_embed(session).embed([f"t{i}" for i in range(65)])
    assert [len(c["json"]["input"]) for c in session.calls] == [32, 32, 1]
    assert session.calls[0]["url"] == "https://api.mistral.ai/v1/embeddings"
    assert session.calls[0]["json"]["model"] == "mistral-embed"
    assert len(vectors) == 65 and vectors[32] == [100.0] and vectors[64] == [200.0]


def test_embed_empty_makes_no_call(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", KEY)
    session = FakeSession()
    assert make_embed(session).embed([]) == []
    assert session.calls == []


def test_embed_wrong_count_raises(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", KEY)
    with pytest.raises(BackendError):
        make_embed(FakeSession(embed_ok(1))).embed(["a", "b"])


def test_embed_retries_429(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", KEY)
    session = FakeSession(FakeResponse(429, text="x"), embed_ok(1))
    assert make_embed(session).embed(["a"]) == [[0.0]]


class Inner:
    name = "inner/m"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, system, user, *, temperature=0.0):
        self.calls += 1
        return self.replies.pop(0)


def test_cache_hit_skips_inner(tmp_path):
    inner = Inner("a", "b")
    cached = CachedChat(inner, tmp_path / "c.jsonl")
    assert cached.name == "inner/m"
    assert cached.complete("s", "u") == "a"
    assert cached.complete("s", "u") == "a"
    assert inner.calls == 1
    assert cached.complete("s", "u2") == "b"


def test_cache_key_covers_temperature_and_system(tmp_path):
    inner = Inner("a", "b", "c")
    cached = CachedChat(inner, tmp_path / "c.jsonl")
    cached.complete("s", "u")
    cached.complete("s", "u", temperature=0.5)
    cached.complete("s2", "u")
    assert inner.calls == 3


def test_cache_persists_across_instances_with_non_ascii(tmp_path):
    path = tmp_path / "sub" / "c.jsonl"
    CachedChat(Inner("réponse ✓"), path).complete("", "café")
    inner = Inner()
    assert CachedChat(inner, path).complete("", "café") == "réponse ✓"
    assert inner.calls == 0


def test_cache_key_depends_on_inner_name(tmp_path):
    path = tmp_path / "c.jsonl"
    CachedChat(Inner("a"), path).complete("", "u")
    other = Inner("b")
    other.name = "other/m"
    assert CachedChat(other, path).complete("", "u") == "b"


@pytest.mark.parametrize("tail", [b'{"key": "abc", "val', "é".encode()[:1], b"not json"])
def test_cache_survives_partial_last_line(tmp_path, tail):
    path = tmp_path / "c.jsonl"
    CachedChat(Inner("a"), path).complete("", "u1")
    with open(path, "ab") as f:
        f.write(tail)
    inner = Inner("b", "c")
    cached = CachedChat(inner, path)
    assert cached.complete("", "u1") == "a"
    assert cached.complete("", "u2") == "b"
    assert CachedChat(Inner(), path).complete("", "u2") == "b"
    assert all(json.loads(line) for line in path.read_text("utf-8").splitlines())


def test_cache_corrupt_middle_line_raises(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_bytes(b'garbage\n{"key": "k", "value": "v"}\n')
    with pytest.raises(ValueError):
        CachedChat(Inner(), path)


def test_failed_call_is_not_cached(tmp_path):
    class Failing(Inner):
        def complete(self, system, user, *, temperature=0.0):
            raise BackendError("down")

    path = tmp_path / "c.jsonl"
    with pytest.raises(BackendError):
        CachedChat(Failing(), path).complete("", "u")
    assert not path.exists() or path.read_bytes() == b""


def test_cache_key_normalises_integer_temperature(tmp_path):
    inner = Inner("a", "b")
    cached = CachedChat(inner, tmp_path / "c.jsonl")
    assert cached.complete("", "u", temperature=0) == "a"
    assert cached.complete("", "u", temperature=0.0) == "a"
    assert inner.calls == 1


def test_extra_body_is_sent_and_recorded_in_name():
    session = FakeSession(chat_ok("hi"))
    backend, _ = make(session, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    backend.complete("", "usr")
    assert session.calls[0]["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert session.calls[0]["json"]["model"] == "m1"
    assert backend.name == 'api.example.com/m1 {"chat_template_kwargs":{"enable_thinking":false}}'
