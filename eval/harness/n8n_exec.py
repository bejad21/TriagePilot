"""Decode the execution records n8n stores in its SQLite database (`execution_data.data`, "flatted" JSON)."""
import json


def unflatten(text: str) -> object:
    """Decode n8n's flatted JSON: strings made of digits inside containers are indexes into the array."""
    arr = json.loads(text)
    if not isinstance(arr, list) or not arr:
        raise ValueError("flatted data must be a non-empty JSON array")
    memo: dict[int, object] = {}

    def resolve(i: int) -> object:
        if i in memo:
            return memo[i]
        if i >= len(arr):
            raise ValueError(f"flatted index {i} out of range")
        value = arr[i]
        # Register the container before filling it so cycles and shared
        # references resolve to the same object.
        if isinstance(value, dict):
            out: dict = {}
            memo[i] = out
            for k, v in value.items():
                out[k] = follow(v)
            return out
        if isinstance(value, list):
            lst: list = []
            memo[i] = lst
            lst.extend(follow(v) for v in value)
            return lst
        memo[i] = value
        return value

    def follow(v: object) -> object:
        if isinstance(v, str) and v.isascii() and v.isdigit():
            return resolve(int(v))
        return v

    return resolve(0)


def node_outputs(run_data: dict, node: str) -> list[dict]:
    """The `json` of every output item of every run of a node, sub-node connections included."""
    outs = []
    for run in run_data.get(node) or []:
        for branches in ((run or {}).get("data") or {}).values():
            for branch in branches or []:
                for item in branch or []:
                    payload = item.get("json") if isinstance(item, dict) else None
                    if isinstance(payload, dict):
                        outs.append(payload)
    return outs
