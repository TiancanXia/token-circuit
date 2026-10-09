"""Flight trails and particle transfer. Visuals consume deterministic simulation."""

from .painting import Painter, BG, GOLD, mix
from .glyphs import character


class FadedGlyph:
    """Tint only character primitives; keep the shared stillworld sprite geometry."""

    def __init__(self, painter, opacity):
        self.painter, self.opacity = painter, opacity

    def rect(self, x, y, w, h, color):
        self.painter.rect(x, y, w, h, mix(BG, color, self.opacity))

    def oval(self, x, y, w, h, color):
        self.painter.oval(x, y, w, h, mix(BG, color, self.opacity))


def flight_trail(p, s):
    trail = s.flight.trail
    for i, (x, y, alpha) in enumerate(trail):
        if i:
            px, py, _ = trail[i - 1]
            p.line([(px, py - 11), (x, y - 11)], mix("#203849", GOLD, alpha * 0.8), 3)
        p.rect(x - 2, y - 13, 3, 3, mix("#1c3445", "#9ad5e1", alpha))
    p.oval(s.x - 18, s.y - 8, 36, 13, mix("#2c4e59", "#827b50", s.flash * 0.6))
    for dx, dy in ((-11, -26), (9, -26), (-11, -4), (9, -4)):
        p.rect(s.x + dx, s.y + dy, 3, 3, GOLD)


def transfer(p, s):
    effect = s.transfer
    progress = effect.progress
    if effect.phase == "dissolve":
        x, y = effect.source
        opacity = max(0.0, 1 - progress / 0.25)
    elif effect.phase == "assemble":
        x, y = effect.target
        opacity = (progress - 0.72) / 0.28
    else:
        opacity = 0.0
    if opacity > 0:
        local = Painter(p.c, p.cn, p.mono, p.scale * 1.4, p.point(x, y))
        character(FadedGlyph(local, opacity), 0, 0)
    colors = (GOLD, "#a0d3df", "#e8e1c7")
    brightness = 1.0 if progress < 0.72 else max(0.0, 1 - (progress - 0.72) / 0.28)
    for x, y, kind in effect.particles():
        size = 2 if kind != 1 else 3
        p.rect(x, y, size, size, mix("#1a3243", colors[kind], brightness))
    if effect.phase == "assemble":
        tx, ty = effect.target
        p.text(tx, ty - 35, s.pending_token, 11, GOLD, anchor="n")
