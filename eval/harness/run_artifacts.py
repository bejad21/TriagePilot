import json
import os
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Iterator

STATUSES = ("answered", "escalated", "rejected", "error")
BEHAVIORS = ("answer", "escalate")


@dataclass
class RunRecord:
    row_id: int
    dataset: str
    split: str
    question: str
    expected_behavior: str
    expected_answer: str | None = None
    gold_sources: list[str] = field(default_factory=list)
    status: str = ""
    answer: str | None = None
    confidence: str | None = None
    cited_source: str | None = None
    retrieved: list[dict] = field(default_factory=list)
    reason: str | None = None
    config: dict = field(default_factory=dict)
    started_at: str = ""
    duration_s: float = 0.0
    error: str | None = None
    context_collected: bool = False


def validate_record(record: RunRecord) -> None:
    """Raise ValueError if the record has an unknown status or behavior, or no question."""
    if record.status not in STATUSES:
        raise ValueError(f"invalid status {record.status!r} for row {record.row_id}")
    if record.expected_behavior not in BEHAVIORS:
        raise ValueError(
            f"invalid expected_behavior {record.expected_behavior!r} for row {record.row_id}")
    if not record.question:
        raise ValueError(f"empty question for row {record.row_id}")


def _parse(line: bytes) -> RunRecord:
    try:
        return RunRecord(**json.loads(line.decode("utf-8")))
    except TypeError as e:
        raise ValueError(f"not a run record: {e}") from e


class RunStore:
    """Append-only JSONL file of RunRecord rows that tolerates a crash mid-write."""

    def __init__(self, path: str | os.PathLike):
        self.path = pathlib.Path(path)

    def append(self, record: RunRecord) -> None:
        """Write one row durably, first dropping any partial line a crash left behind."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._repair_tail()
        line = json.dumps(asdict(record), ensure_ascii=False) + "\n"
        with open(self.path, "ab") as f:
            f.write(line.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())

    def _repair_tail(self) -> None:
        # appending after a partial line would glue two rows into one invalid line
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        if not data or data.endswith(b"\n"):
            return
        cut = data.rfind(b"\n") + 1
        try:
            _parse(data[cut:])
        except ValueError:
            with open(self.path, "r+b") as f:
                f.truncate(cut)
            return
        with open(self.path, "ab") as f:
            f.write(b"\n")

    def __iter__(self) -> Iterator[RunRecord]:
        """Yield every stored row in file order, duplicates included."""
        if not self.path.exists():
            return
        # Decode per line: a crash can cut the last line inside a multibyte character.
        lines = self.path.read_bytes().split(b"\n")
        numbered = [(i, ln) for i, ln in enumerate(lines, 1) if ln.strip()]
        for pos, (lineno, line) in enumerate(numbered):
            try:
                yield _parse(line)
            except ValueError as e:
                if pos == len(numbered) - 1:
                    return
                raise ValueError(f"{self.path}: invalid record on line {lineno}: {e}") from e

    def load(self) -> list[RunRecord]:
        """Rows keyed by row_id, last occurrence wins, in first-seen order."""
        by_id: dict[int, RunRecord] = {}
        for rec in self:
            by_id[rec.row_id] = rec
        return list(by_id.values())

    def done_ids(self) -> set[int]:
        return {rec.row_id for rec in self}
