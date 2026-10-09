"""Data definitions for the model village and grouped module interiors."""

from dataclasses import dataclass


@dataclass(frozen=True)
class House:
    key: str
    title: str
    x: float
    y: float


@dataclass(frozen=True)
class Group:
    house: str
    block: int
    indices: tuple


@dataclass(frozen=True)
class Checkpoint:
    index: int
    x: float
    y: float


HOUSE_DATA = (
    ("embedding", "Embedding", 110, 210),
    ("norm_attn", "RMSNorm", 314, 210),
    ("attention", "Attention", 500, 210),
    ("residual_attn", "Residual +", 680, 210),
    ("norm_ffn", "RMSNorm", 680, 366),
    ("ffn", "FFN / MoE", 500, 366),
    ("residual_ffn", "Residual +", 314, 366),
    ("norm_final", "Final RMS", 314, 520),
    ("unembedding", "Unembedding", 500, 520),
    ("output", "Sampling", 680, 520),
)
KIND_HOUSE = {
    "qkv": "attention",
    "rope": "attention",
    "attention": "attention",
    "project": "attention",
    "up": "ffn",
    "activation": "ffn",
    "down": "ffn",
    "router": "ffn",
    "experts": "ffn",
    "combine": "ffn",
    "softmax": "output",
    "sample": "output",
}
EXIT = (708.0, 333.0)
MICRO_SIZE = (760.0, 410.0)


def houses():
    return tuple(House(*row) for row in HOUSE_DATA)


def doorway(key):
    h = next(h for h in houses() if h.key == key)
    return float(h.x), float(h.y + 35)


def road_edges():
    places = houses()
    return tuple(zip(places, places[1:])) + ((places[6], places[1]),)


def on_platform(h, x, y):
    return abs(x - h.x) <= 76 and abs(y - h.y - 20) <= 66


def on_road(a, b, x, y):
    ax, ay, bx, by = a.x, a.y + 35, b.x, b.y + 35
    return (
        ay == by
        and min(ax, bx) <= x <= max(ax, bx)
        and abs(y - ay) <= 13
        or ax == bx
        and min(ay, by) <= y <= max(ay, by)
        and abs(x - ax) <= 13
    )


def approach_door(position):
    """Approach a door on the occupied platform/road, never across a gap."""
    x, y = position
    candidates = [h for h in houses() if on_platform(h, x, y)]
    if not candidates:
        candidates = [h for a, b in road_edges() if on_road(a, b, x, y) for h in (a, b)]
    if not candidates:
        raise ValueError("Flight must start on the map")
    return min(candidates, key=lambda h: (h.x - x) ** 2 + (h.y + 35 - y) ** 2).key


def group_for(stages, index):
    stage = stages[index]
    key = KIND_HOUSE.get(stage.kind, stage.kind)
    indices = tuple(
        i
        for i, other in enumerate(stages)
        if other.block == stage.block and KIND_HOUSE.get(other.kind, other.kind) == key
    )
    return Group(key, stage.block, indices)


def all_groups(stages):
    groups = []
    index = 0
    while index < len(stages):
        group = group_for(stages, index)
        groups.append(group)
        index = group.indices[-1] + 1
    return groups


def checkpoints(group):
    count = len(group.indices)
    xs = [380.0] if count == 1 else [110 + i * 540 / (count - 1) for i in range(count)]
    return tuple(Checkpoint(index, x, 275.0) for index, x in zip(group.indices, xs))


def macro_floor(x, y):
    places = houses()
    if any(on_platform(h, x, y) for h in places):
        return True
    return any(on_road(a, b, x, y) for a, b in road_edges())


def micro_floor(x, y):
    return 42 <= x <= 745 and 230 <= y <= 371
