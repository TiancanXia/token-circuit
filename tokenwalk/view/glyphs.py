"""Shared stillworld pixel character. No network-specific legacy symbols."""

import math


def character(p, x, y, moving=False, phase=0.0, facing=(0, 1), ghost=False):
    pale, head, scarf = (
        ("#829aab", "#b0c7ce", "#89b2c0")
        if ghost
        else ("#cfcebb", "#ece7cf", "#d9bb78")
    )
    p.oval(x - 6, y - 2, 12, 4, "#1c3038")
    step = 1 if moving and math.sin(phase * 17) > 0 else 0
    p.rect(x - 3, y - 4 - step, 2, 4, pale)
    p.rect(x + 1, y - 3 + step, 2, 3, pale)
    p.rect(x - 4, y - 11, 8, 7, pale)
    p.rect(x - 3, y - 18, 6, 7, head)
    p.rect(x - 3, y - 18, 6, 2, "#566565")
    p.rect(x - 4, y - 11, 8, 2, scarf)
    p.rect(x + 3, y - 9, 3, 2, scarf)
    fx, fy = facing
    if fx:
        p.rect(x + (2 if fx > 0 else -3), y - 14, 1, 1, "#334c50")
    elif fy > 0:
        for offset in (-2, 1):
            p.rect(x + offset, y - 14, 1, 1, "#334c50")
