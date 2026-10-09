"""Enterable neural devices and playable circuit floors. Drawing only."""

import math
from .glyphs import character
from ..simulation.maps import houses, road_edges
from .painting import Painter, GOLD, TEXT, MUTED, mix
from .effects import flight_trail


def actor(p, s):
    if s.scene == "birth":
        return
    if s.scene == "flight":
        flight_trail(p, s)
    local = Painter(p.c, p.cn, p.mono, p.scale * 1.4, p.point(s.x, s.y))
    character(local, 0, 0, s.moving, s.walk_time, s.facing)
    p.fit_text(s.x, s.y - 32, s.token, 70, 9, GOLD, anchor="n")


def house(p, h, glow, expected, title, time=0.0):
    x, y = h.x, h.y
    glow *= 0.88 + 0.12 * math.sin(time * 3)
    edge = mix("#52788a", GOLD, glow)
    fill = mix("#223c4b", "#435352", glow)
    core = mix("#729ba8", "#f3d490", glow)
    base = [
        (x - 57, y + 24),
        (x - 40, y + 13),
        (x + 40, y + 13),
        (x + 57, y + 24),
        (x + 40, y + 47),
        (x - 40, y + 47),
    ]
    p.poly(base, "#182f3d")
    p.line(base + [base[0]], edge)
    p.line([(x - 45, y + 36), (x + 45, y + 36)], "#365967")
    if h.key == "embedding":
        for row in range(3):
            for col in range(3):
                px, py = x - 24 + col * 17, y - 38 + row * 17
                p.poly(
                    [(px, py), (px + 12, py - 6), (px + 23, py), (px + 11, py + 6)],
                    core if row == col else fill,
                )
                p.line(
                    [
                        (px, py),
                        (px, py + 7),
                        (px + 11, py + 13),
                        (px + 23, py + 7),
                        (px + 23, py),
                    ],
                    edge,
                )
    elif h.key == "attention":
        for dx, height in ((-31, 39), (0, 55), (31, 39)):
            p.poly(
                [
                    (x + dx - 8, y + 9),
                    (x + dx - 8, y - height),
                    (x + dx, y - height - 6),
                    (x + dx + 8, y - height),
                    (x + dx + 8, y + 9),
                ],
                fill,
            )
            p.line([(x + dx, y - height - 3), (x + dx, y + 6)], core, 2)
            p.rect(x + dx - 4, y - height + 7, 8, 7, core)
        p.line(
            [(x - 31, y - 21), (x, y - 37), (x + 31, y - 21), (x - 31, y - 21)], edge
        )
    elif h.key == "ffn":
        for layer, half in enumerate((20, 42, 30)):
            py = y - 33 + layer * 19
            points = [(x - half, py), (x, py - 12), (x + half, py), (x, py + 12)]
            p.poly(points, fill)
            p.line(points + [points[0]], core if layer == 1 else edge)
        p.line([(x, y - 45), (x, y + 17)], core)
    elif h.key.startswith("norm"):
        for half, py in ((34, y - 40), (26, y - 20), (18, y)):
            p.oval(x - half, py, half * 2, 18, edge)
            p.oval(x - half + 3, py + 3, half * 2 - 6, 12, "#102736")
        p.line([(x, y - 41), (x, y + 12)], core, 2)
    elif h.key.startswith("residual"):
        for dx in (-14, 14):
            p.oval(x + dx - 24, y - 34, 48, 45, edge)
            p.oval(x + dx - 21, y - 31, 42, 39, "#102736")
        p.line([(x - 8, y - 12), (x + 8, y - 12)], core, 2)
        p.line([(x, y - 20), (x, y - 4)], core, 2)
    else:
        for i in range(5):
            angle = (i * 30 + 210) * math.pi / 180
            px, py = x + math.cos(angle) * 45, y + math.sin(angle) * 45
            p.line([(x, y), (px, py)], edge)
            p.rect(px - 3, py - 3, 6, 6, core)
        p.poly([(x, y - 23), (x + 18, y - 5), (x, y + 13), (x - 18, y - 5)], fill)
        p.line(
            [(x, y - 17), (x + 12, y - 5), (x, y + 7), (x - 12, y - 5), (x, y - 17)],
            core,
        )
    # A lit portal stays at the same walkable doorway as the original houses.
    p.rect(x - 10, y + 15, 20, 27, "#091a27")
    p.line(
        [(x - 10, y + 40), (x - 10, y + 15), (x + 10, y + 15), (x + 10, y + 40)],
        core,
        2,
    )
    p.rect(x - 15, y + 43, 30, 3, edge)
    p.text(x, y - 65, title, 13, TEXT, anchor="n", mono=True)
    if expected:
        p.rect(x - 5, y + 54, 10, 3, GOLD)
        p.text(x, y + 62, "E", 9, GOLD, anchor="n")


def macro(p, s):
    places = houses()
    p.rect(232, 107, 536, 337, "#132a38")
    p.line(
        [(232, 444), (232, 107), (768, 107), (768, 444), (232, 444)],
        "#426b7d",
        1,
        (4, 3),
    )
    config = s.forward.config
    block = min(s.block, len(config.blocks) - 1)
    style = "Dense FFN" if config.blocks[block] == "dense" else "MoE FFN"
    p.text(253, 121, f"Transformer Block × {len(config.blocks)}", 11, GOLD, mono=True)
    p.text(
        751,
        121,
        p.tr("block", n=block + 1, total=len(config.blocks), style=style),
        9,
        MUTED,
        anchor="ne",
    )
    for a, b in road_edges():
        points = [(a.x, a.y + 35), (b.x, b.y + 35)]
        p.line(points, "#1b3446", 25)
        p.line(points, "#2b5061", 19)
        p.line(points, "#7294a1", 1)
    # Residual signals are distinct from roads.
    p.line([(314, 186), (314, 157), (680, 157), (680, 186)], "#7da8b7", 1, (3, 4))
    p.line([(680, 348), (680, 318), (314, 318), (314, 348)], "#7da8b7", 1, (3, 4))
    expected = s.expected_group
    for h in places:
        title = (
            ("FFN" if style == "Dense FFN" else "MoE")
            if h.key == "ffn"
            else "Output"
            if h.key == "output" and s.guided
            else h.title
        )
        if h.key == "attention":
            title += " / " + config.attention_mode
        house(
            p,
            h,
            s.glows[h.key],
            bool(expected and expected.house == h.key),
            title,
            s.time,
        )
    actor(p, s)
