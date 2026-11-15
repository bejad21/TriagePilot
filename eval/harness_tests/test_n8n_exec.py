import json

import pytest

from harness.n8n_exec import node_outputs, unflatten


def test_unflatten_resolves_digit_strings_as_indexes():
    flat = [{"name": "1", "items": "2"}, "Answer", ["3", "3"], {"n": 5}]
    out = unflatten(json.dumps(flat))
    assert out["name"] == "Answer"
    assert out["items"][0] is out["items"][1] and out["items"][0] == {"n": 5}


def test_unflatten_keeps_cycles_as_the_same_object():
    out = unflatten(json.dumps([{"self": "0", "x": 1}]))
    assert out["self"] is out and out["x"] == 1


def test_unflatten_rejects_empty_and_non_array_input():
    for bad in ("[]", '{"a": 1}'):
        with pytest.raises(ValueError, match="non-empty JSON array"):
            unflatten(bad)


def test_unflatten_rejects_an_index_past_the_end():
    with pytest.raises(ValueError, match="out of range"):
        unflatten(json.dumps([{"a": "9"}]))


def test_node_outputs_collects_every_run_and_connection():
    run_data = {"Answer": [
        {"data": {"main": [[{"json": {"x": 1}}, {"json": {"x": 2}}]], "ai_tool": [[{"json": {"y": 3}}]]}},
        {"data": {"main": [[{"json": {"x": 4}}, {"binary": {}}, None]]}},
    ]}
    assert node_outputs(run_data, "Answer") == [{"x": 1}, {"x": 2}, {"y": 3}, {"x": 4}]


def test_node_outputs_is_empty_for_a_node_that_did_not_run():
    assert node_outputs({}, "Answer") == [] and node_outputs({"Answer": None}, "Answer") == []
