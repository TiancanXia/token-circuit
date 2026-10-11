"""Shared walkable room floor for modules and nested interiors."""

from ..simulation.maps import EXIT
from ..simulation.rooms import ROOMS
from .map_view import actor
from .painting import GOLD, TEXT, MUTED


def draw_room(p, s):
    p.poly(
        [
            (43, 224),
            (735, 224),
            (752, 242),
            (752, 357),
            (735, 375),
            (43, 375),
            (26, 357),
            (26, 242),
        ],
        "#182f40",
    )
    p.line([(43, 224), (735, 224), (752, 242)], "#426e82")
    for x in range(70, 730, 42):
        p.line([(x, 234), (x, 365)], "#203d4b")
    p.line([(70, 310), (700, 310)], "#365969", 14)
    p.line([(70, 310), (700, 310)], "#7b9aa5")
    for cp in s.room_checkpoints:
        complete = s.checkpoint_complete(cp.index)
        near = s.near_checkpoint
        focus = bool(near and near.index == cp.index)
        color = GOLD if focus else "#94a696" if complete else "#738b88"
        p.poly(
            [
                (cp.x - 38, cp.y - 18),
                (cp.x + 29, cp.y - 18),
                (cp.x + 38, cp.y - 9),
                (cp.x + 38, cp.y + 6),
                (cp.x - 38, cp.y + 6),
            ],
            "#203c4d",
        )
        p.rect(cp.x - 38, cp.y - 21, 76, 4, color)
        for dx in (-23, -7, 9, 25):
            p.rect(cp.x + dx, cp.y - 12, 6, 8, "#547d8b")
        p.line([(cp.x - 29, cp.y + 9), (cp.x + 29, cp.y + 9)], color)
        title = s.checkpoint_title(cp.index)
        readable = s.scene == "detail" and bool(s.rooms.active.spec.label_keys)
        p.fit_text(
            cp.x,
            cp.y - 51,
            title,
            100 if readable else 115,
            17 if readable else 14,
            TEXT,
            anchor="n",
            mono=not readable,
        )
        child = s.scene == "micro" and s.forward.stages[cp.index].kind in ROOMS
        p.text(
            cp.x,
            cp.y + 30,
            p.tr("detail_enter" if child else "processed" if complete else "process"),
            11,
            color,
            anchor="n",
        )
    ex, ey = EXIT
    p.rect(ex - 16, ey - 38, 32, 42, "#1c303a")
    p.rect(ex - 11, ey - 33, 22, 37, GOLD if s.group_complete else "#4d6465")
    p.text(
        ex,
        ey + 10,
        p.tr("exit" if s.group_complete else "exit_wait"),
        8,
        GOLD if s.group_complete else MUTED,
        anchor="n",
    )
    actor(p, s)
