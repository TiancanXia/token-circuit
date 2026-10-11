"""Registered child rooms and visits, independent of graphics and input bindings."""

from dataclasses import dataclass, field

from .maps import Checkpoint


@dataclass(frozen=True)
class RoomSpec:
    kind: str
    nodes: tuple
    label_keys: tuple = ()
    operation_indices: tuple = ()

    def operation_index(self, index):
        return self.operation_indices[index] if self.operation_indices else index

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
    kind: RoomSpec(
        kind,
        nodes,
        tuple(f"step.{kind}.{i}" for i in range(len(nodes))),
    )
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
    """Specialize layouts and omit a decode mask only when every query is visible."""
    base = ROOMS[result.stage.kind]
    nodes = list(base.nodes)
    labels = list(base.label_keys)
    operations = ()
    if base.kind == "qkv":
        nodes[4] = config.attention_mode + " heads"
    elif (
        base.kind == "attention"
        and result.details.get("phase") == "decode"
        and not any(any(trace["mask_applied"]) for trace in result.details["traces"])
    ):
        operations = tuple(i for i in range(len(nodes)) if i != 2)
        nodes = [nodes[i] for i in operations]
        labels = [labels[i] for i in operations]
    return RoomSpec(base.kind, tuple(nodes), tuple(labels), operations)


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
            visit = Visit(index, spec, result, head=session.attention_probe.head)
            self.visits[index] = visit
        spec = visit.spec
        visit.head = session.attention_probe.head
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
            session.say(
                session.tr("detail_step", title=session.checkpoint_title(cp.index))
            )
        return True
