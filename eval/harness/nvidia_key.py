"""The NVIDIA API key from the environment (NVIDIA_API_KEY)."""
import os


def api_key():
    """Return the key; raises KeyError if it is not set."""
    try:
        return os.environ["NVIDIA_API_KEY"]
    except KeyError:
        raise KeyError("NVIDIA_API_KEY is not set") from None
