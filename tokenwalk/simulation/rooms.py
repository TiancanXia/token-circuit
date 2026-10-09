"""Registered child rooms and visits, independent of graphics and input bindings."""

from dataclasses import dataclass, field
from .maps import Checkpoint


@dataclass(frozen=True)
class RoomSpec:
    kind: str
    nodes: tuple

    def checkpoints(self):
        count = len(self.nodes)
        return tuple(
            Checkpoint(
                i, 380.0 if count == 1 else 110.0 + 540.0 * i / (count - 1), 275.0
            )
            for i in range(count)
        )


# Add a specification and a view drawer to introduce another module interior.
ROOMS = {
    kind: RoomSpec(kind, nodes)
    for kind, nodes in (
        ("qkv", ("X", "Q = X Wqᵀ", "K = X Wkᵀ", "V = X Wvᵀ", "Heads")),
        ("rope", ("Position", "Rotate Q/K", "Ready K/V")),
        (
            "attention",
            ("Read K/V", "Q Kᵀ / √d", "Causal mask", "Softmax", "Σ p V", "Concat"),
        ),
        ("project", ("Input", "W_o", "Output")),
    )
}


def specification(result, config):
    """Refine labels from real diagnostics; checkpoint count/order stays stable."""
    base = ROOMS[result.stage.kind]
    nodes = list(base.nodes)
    if base.kind == "qkv":
        nodes[4] = config.attention_mode + " heads"
    elif base.kind == "attention" and not any(
        result.details["traces"][0]["mask_applied"]
    ):
        nodes[2] = "All visible"
    return RoomSpec(base.kind, tuple(nodes))


@dataclass
class Visit:
    index: int
    spec: RoomSpec
    result: object
    progress: int = 0
    head: int = 0

    @property
    def complete(self):
        return self.progress == len(self.spec.nodes)


@dataclass
class Rooms:
    visits: dict = field(default_factory=dict)
    parents: list = field(default_factory=list)
    active: Visit | None = None

    def enter(self, session, index):
        spec = ROOMS.get(session.forward.stages[index].kind)
        if spec is None or index > session.forward.next_index:
            return False
        visit = self.visits.get(index)
        if visit is None:
            result = session.forward.preview(index)
            spec = specification(result, session.forward.config)
            visit = Visit(index, spec, result)
            self.visits[index] = visit
        if index < session.forward.next_index:
            visit.progress = len(spec.nodes)
        self.parents.append((session.scene, session.x, session.y))
        self.active = visit
        session.scene = "detail"
        cp = spec.checkpoints()[min(visit.progress, len(spec.nodes) - 1)]
        session.x, session.y = cp.x, cp.y + 22
        session.moving = False
        session.say(session.tr("detail_hint"))
        return True

    def back(self, session):
        if not self.parents:
            return False
        session.scene, session.x, session.y = self.parents.pop()
        self.active = None
        session.moving = False
        session.say(session.tr("room_hint"))
        return True

    def interact(self, session, cp):
        visit = self.active
        if cp.index != visit.progress:
            session.say(
                session.tr(
                    "already_done" if cp.index < visit.progress else "detail_order"
                )
            )
            return False
        visit.progress += 1
        if visit.complete and visit.index == session.forward.next_index:
            session._execute_step(visit.index)
            visit.result = session.forward.results[visit.index]
        else:
            session.say(session.tr("detail_step", title=visit.spec.nodes[cp.index]))
        return True
