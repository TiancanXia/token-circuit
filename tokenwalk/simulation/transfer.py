"""Deterministic dissolve/transport/assemble geometry, independent of the UI."""

import math
import random


def ease(t):
    return t * t * (3 - 2 * t)


class Transfer:
    duration = 1.4

    def __init__(self, source, target, seed):
        self.source, self.target = source, target
        self.elapsed = 0.0
        rng = random.Random(seed)
        self.offsets = tuple(
            (
                rng.uniform(-7, 7),
                rng.uniform(-23, -3),
                rng.uniform(0, math.tau),
                rng.uniform(15, 45),
            )
            for _ in range(42)
        )

    @property
    def progress(self):
        return min(1.0, self.elapsed / self.duration)

    @property
    def phase(self):
        return (
            "dissolve"
            if self.progress < 0.25
            else "drift"
            if self.progress < 0.72
            else "assemble"
        )

    def particles(self):
        t = self.progress
        sx, sy = self.source
        tx, ty = self.target
        points = []
        for i, (ox, oy, angle, radius) in enumerate(self.offsets):
            if t < 0.25:
                spread = ease(t / 0.25)
                x, y = (
                    sx + ox + math.cos(angle) * radius * spread,
                    sy + oy + math.sin(angle) * radius * spread,
                )
            elif t < 0.72:
                u = ease((t - 0.25) / 0.47)
                a, b, c, d = (
                    (1 - u) ** 3,
                    3 * (1 - u) ** 2 * u,
                    3 * (1 - u) * u * u,
                    u**3,
                )
                x = a * sx + b * 740 + c * 240 + d * tx
                y = a * sy + b * 170 + c * 135 + d * ty
                scatter = (1 - u) * radius + u * 28
                x += ox + math.cos(angle + u * math.tau) * scatter
                y += oy + math.sin(angle + u * math.tau) * scatter
            else:
                remaining = 1 - ease((t - 0.72) / 0.28)
                x, y = (
                    tx + ox + math.cos(angle) * 28 * remaining,
                    ty + oy + math.sin(angle) * 28 * remaining,
                )
            points.append((x, y, i % 3))
        return tuple(points)
