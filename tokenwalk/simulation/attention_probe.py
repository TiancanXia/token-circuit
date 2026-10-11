"""Read-only selections over actual attention trace snapshots."""

from dataclasses import dataclass


@dataclass
class AttentionProbe:
    row: int = -1
    key: int = 0
    head: int = 0

    def indices(self, data):
        rows = len(data["positions"])
        row = rows - 1 if self.row < 0 else min(self.row, rows - 1)
        key = max(0, min(self.key, len(data["k_matrix"]) - 1))
        return row, key


@dataclass(frozen=True)
class KeyInspection:
    position: int
    kv_head: int
    score: float
    probability: float
    value: tuple
    contribution: tuple
    output: tuple
    masked: bool
    cached: bool


def inspect_key(data, config, head, row, key):
    trace = data["traces"][head]
    probability = trace["weights"][row][key]
    start = trace["kv_head"] * config.head_dim
    value = tuple(data["v_matrix"][key][start : start + config.head_dim])
    return KeyInspection(
        data["positions"][row],
        trace["kv_head"],
        trace["scores"][row][key],
        probability,
        value,
        tuple(probability * x for x in value),
        tuple(trace["outputs"][row]),
        key > data["positions"][row],
        key < data["reused"],
    )
