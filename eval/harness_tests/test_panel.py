import pytest

from harness import panel


def test_current_panel_has_no_mistral_judge():
    assert panel.PANEL == ("nemotron", "glm45", "laguna")
    assert "mistral" not in panel.PANEL
    assert panel.OLD_PANEL == ("mistral", "nemotron", "glm45")


@pytest.mark.parametrize("scores,expected", [
    ((5, 4, 1), True),
    ((4, 4, 4), True),
    ((5, 3, 3), False),
    ((3, 3, 3), False),
    ((4, 3, None), False),
    ((4, 5, None), True),
    ((None, None, None), False),
])
def test_is_good_needs_two_votes_of_four_or_more(scores, expected):
    assert panel.is_good(scores) is expected


@pytest.mark.parametrize("name,model,key_env", [
    ("nemotron", "nvidia/nemotron-3.5-super-vl-120b-a12b", "NVIDIA_API_KEY"),
    ("glm45", "glm-4.5-flash", "ZAI_API_KEY"),
    ("laguna", "poolside/laguna-xs-2.1", "NVIDIA_API_KEY"),
])
def test_build_returns_a_backend_without_calling_the_network(name, model, key_env):
    backend = panel.build(name)
    assert backend.model == model
    assert backend.api_key_env == key_env


def test_build_rejects_an_unknown_judge():
    with pytest.raises(KeyError):
        panel.build("gpt")
