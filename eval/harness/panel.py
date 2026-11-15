"""Judge panel for Answer Correctness.

PANEL is Nemotron-3.5-super (NVIDIA), GLM-4.5-flash (Z.ai) and Laguna-xs-2.1
(Poolside, hosted on NVIDIA). None is from the family of the answer model (Mistral).
OLD_PANEL is the earlier panel, kept to re-score saved grades.

An answer is good when at least two of the three judges give it 4 or more; a missing
score counts as not good. Constructing a backend sends nothing; the keys
(NVIDIA_API_KEY, ZAI_API_KEY) are read at call time.
"""
from harness.backends import OpenAICompatChat

PANEL = ("nemotron", "glm45", "laguna")
OLD_PANEL = ("mistral", "nemotron", "glm45")
GOOD_AT = 4
VOTES_NEEDED = 2

_NVIDIA = "https://integrate.api.nvidia.com/v1"
_ZAI = "https://api.z.ai/api/paas/v4"


def build(name: str):
    if name == "nemotron":
        return OpenAICompatChat("nvidia/nemotron-3.5-super-vl-120b-a12b", "NVIDIA_API_KEY", _NVIDIA,
                                extra_body={"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 400})
    if name == "glm45":
        return OpenAICompatChat("glm-4.5-flash", "ZAI_API_KEY", _ZAI,
                                extra_body={"thinking": {"type": "disabled"}, "max_tokens": 400})
    if name == "laguna":
        return OpenAICompatChat("poolside/laguna-xs-2.1", "NVIDIA_API_KEY", _NVIDIA, extra_body={"max_tokens": 2500})
    raise KeyError(f"unknown judge {name!r}")


def is_good(scores) -> bool:
    """True when at least VOTES_NEEDED of the 1-to-5 scores reach GOOD_AT; None counts as not good."""
    return sum(1 for s in scores if s is not None and s >= GOOD_AT) >= VOTES_NEEDED
