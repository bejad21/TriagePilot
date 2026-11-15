"""Judge selection shared by the `score` subcommand and `validate_judge`."""
import json
import os
from urllib.parse import urlsplit

from harness.backends import MistralChat, OpenAICompatChat


def unrecordable_url_error(url: str, label: str) -> str | None:
    """Why `url` cannot be recorded (user info or query string), or None. Never echoes the URL."""
    try:
        netloc = urlsplit(url).netloc
    except ValueError:
        return f"the {label} cannot be parsed"
    if "@" in netloc or "?" in url:
        return (f"the {label} must not hold user info (user:pass@) or a "
                "query string; it would be recorded")
    return None


def judge_args_error(base_url: str | None, key_env: str | None) -> str | None:
    """Problem with the --judge-base-url / --judge-key-env pair, or None."""
    if bool(base_url) != bool(key_env):
        return "--judge-base-url and --judge-key-env go together"
    if base_url:
        return unrecordable_url_error(base_url, "judge base URL")
    return None


def judge_key_env(base_url: str | None, key_env: str | None) -> str:
    """Env var that holds the judge key: `key_env` with a custom base URL, else MISTRAL_API_KEY."""
    return key_env if base_url else "MISTRAL_API_KEY"


def parse_extra_body(text: str | None) -> dict | None:
    """The --judge-extra-body JSON object, or None; ValueError if it is not an object."""
    if not text:
        return None
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("--judge-extra-body must be a JSON object")
    return value


def build_judge(model: str, base_url: str | None, key_env: str | None, *, session=None,
                extra_body: dict | None = None):
    """The judge chat backend; constructing it sends nothing."""
    if base_url:
        return OpenAICompatChat(model, key_env, base_url, session=session,
                                extra_body=extra_body)
    return MistralChat(model, session=session)


def missing_key_env(base_url: str | None, key_env: str | None) -> str | None:
    name = judge_key_env(base_url, key_env)
    return None if os.environ.get(name) else name
