"""HTTP chat and embedding backends for the judges, plus a disk cache for chat.

Retries cost API quota, so only transient failures are retried: HTTP 429, 500,
502, 503, 504 and connection or timeout errors. Any other status fails at once.
The API key is read at call time and never appears in an exception message.
"""
import hashlib
import json
import os
import pathlib
import time
from typing import Callable
from urllib.parse import urlparse

import requests

from harness.judges import ChatBackend

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_BODY_LIMIT = 300
_EMBED_BATCH = 32
_BACKOFF_BASE_S = 1.0
_BACKOFF_CAP_S = 60.0


class BackendError(RuntimeError):
    """A backend call failed or returned nothing usable."""


class _Client:
    def __init__(self, model: str, api_key_env: str, base_url: str, *,
                 timeout: float = 120.0, max_retries: int = 5, session=None,
                 sleep: Callable[[float], None] = time.sleep):
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session if session is not None else requests.Session()
        self.sleep = sleep
        self.name = f"{urlparse(self.base_url).netloc}/{model}"

    def _post(self, path: str, payload: dict) -> dict:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise BackendError(f"missing key env var {self.api_key_env}")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    url, json=payload, headers=headers, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                # The exception text is dropped: it can echo request details.
                failure = f"{type(e).__name__} calling {url}"
            else:
                status = response.status_code
                if status == 200:
                    return self._json(response)
                failure = f"HTTP {status} from {url}: {self._excerpt(response.text)}"
                if status not in _RETRY_STATUSES:
                    raise BackendError(failure)
            if attempt == self.max_retries:
                raise BackendError(f"{failure} (after {attempt + 1} attempts)")
            self.sleep(min(_BACKOFF_BASE_S * 2 ** attempt, _BACKOFF_CAP_S))
        raise AssertionError("unreachable")

    def _excerpt(self, text: str) -> str:
        """Truncated text for error messages, with the API key removed first."""
        key = os.environ.get(self.api_key_env)
        if key:
            text = text.replace(f"Bearer {key}", "Bearer ***").replace(key, "***")
        return text[:_BODY_LIMIT]

    def _json(self, response) -> dict:
        try:
            body = response.json()
        except ValueError:
            raise BackendError(f"response is not JSON: {self._excerpt(response.text)!r}") from None
        if not isinstance(body, dict):
            raise BackendError(f"unexpected response: {self._excerpt(response.text)!r}")
        return body


class OpenAICompatChat(_Client):
    """Chat completions against any OpenAI-compatible endpoint.

    `extra_body` fields are merged into every request (e.g. to switch a model's
    thinking off) and appear in `name`, so they are recorded with the judge.
    """

    def __init__(self, *args, extra_body: dict | None = None, **kw):
        super().__init__(*args, **kw)
        self.extra_body = dict(extra_body or {})
        if self.extra_body:
            self.name += " " + json.dumps(self.extra_body, sort_keys=True, separators=(",", ":"))

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        """Return the reply text; raise BackendError on failure or empty content."""
        messages = [{"role": "user", "content": user}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        body = self._post("/chat/completions", {
            **self.extra_body, "model": self.model, "temperature": temperature,
            "messages": messages})
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise BackendError(
                f"response has no message content: {self._excerpt(str(body))}") from None
        if not isinstance(content, str) or not content.strip():
            raise BackendError("response message content is empty")
        return content


class MistralChat(OpenAICompatChat):
    def __init__(self, model: str = "mistral-small-latest", **kw):
        super().__init__(model, "MISTRAL_API_KEY", "https://api.mistral.ai/v1", **kw)


class MistralEmbed(_Client):
    """Mistral embeddings, at most 32 texts per request."""

    def __init__(self, model: str = "mistral-embed", **kw):
        super().__init__(model, "MISTRAL_API_KEY", "https://api.mistral.ai/v1", **kw)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text, in order."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH):
            batch = texts[i:i + _EMBED_BATCH]
            body = self._post("/embeddings", {"model": self.model, "input": batch})
            vectors.extend(self._vectors(body, len(batch)))
        return vectors

    def _vectors(self, body: dict, expected: int) -> list[list[float]]:
        data = body.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise BackendError(f"expected {expected} embeddings, got {self._excerpt(str(data))}")
        try:
            ordered = sorted(data, key=lambda item: item["index"])
            return [item["embedding"] for item in ordered]
        except (KeyError, TypeError):
            raise BackendError("embedding response items lack index or embedding") from None


def _parse_entry(line: bytes) -> tuple[str, str]:
    try:
        entry = json.loads(line.decode("utf-8"))
        key, value = entry["key"], entry["value"]
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"not a cache entry: {e}") from e
    if not isinstance(key, str) or not isinstance(value, str):
        raise ValueError("cache entry key and value must be strings")
    return key, value


class CachedChat:
    """Disk-cached wrapper around a ChatBackend; hits make no inner call."""

    def __init__(self, inner: ChatBackend, cache_path: str | os.PathLike):
        self.inner = inner
        self.name = inner.name
        self.path = pathlib.Path(cache_path)
        self._entries = self._load()

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        key = hashlib.sha256(json.dumps(
            [self.inner.name, system, user, float(temperature)]).encode("utf-8")).hexdigest()
        if key in self._entries:
            return self._entries[key]
        value = self.inner.complete(system, user, temperature=temperature)
        self._append(key, value)
        self._entries[key] = value
        return value

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        # Decode per line: a crash can cut the last line inside a multibyte character.
        lines = [ln for ln in self.path.read_bytes().split(b"\n") if ln.strip()]
        entries: dict[str, str] = {}
        for pos, line in enumerate(lines):
            try:
                key, value = _parse_entry(line)
            except ValueError as e:
                if pos == len(lines) - 1:
                    break
                raise ValueError(f"{self.path}: invalid cache line {pos + 1}: {e}") from e
            entries[key] = value
        return entries

    def _append(self, key: str, value: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._repair_tail()
        line = json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n"
        with open(self.path, "ab") as f:
            f.write(line.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())

    def _repair_tail(self) -> None:
        # Appending after a partial line would glue two entries together.
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        if not data or data.endswith(b"\n"):
            return
        cut = data.rfind(b"\n") + 1
        try:
            _parse_entry(data[cut:])
        except ValueError:
            with open(self.path, "r+b") as f:
                f.truncate(cut)
            return
        with open(self.path, "ab") as f:
            f.write(b"\n")
