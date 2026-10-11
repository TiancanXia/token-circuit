"""Reusable numeric glyphs; no scene transitions."""

from .painting import GOLD, TEXT, MUTED, mix


def result_for(s, index):
    return s.forward.results[index] if index < len(s.forward.results) else None


def bars(p, x, y, values, width=160, height=31, **kw):
    if not values:
        p.text(x, y, p.tr("wait_compute"), 8, MUTED)
        return
    maximum = max(0.01, max(abs(v) for v in values))
    step = width / len(values)
    for i, v in enumerate(values):
        h = abs(v) / maximum * height
        p.rect(
            x + i * step + 1,
            y + height - h,
            max(1, step - 3),
            h,
            GOLD if v >= 0 else "#89b3c3",
            **kw,
        )


def matrix(p, x, y, rows, label, s, width=166, positions=None):
    if not rows:
        p.text(x, y, p.tr("project_hint", label=label), 11, TEXT, mono=True)
        p.rect(x, y + 29, width - 5, 94, "#233947")
        p.text(x + 10, y + 49, p.tr("matrix_wait"), 9, MUTED)
        return
    n, d = len(rows), len(rows[0])
    p.text(x, y, f"{label}  {n}×{d}", 11, GOLD, mono=True)
    shown = rows[-5:]
    cell = min(15, (width - 35) / d)
    maximum = max(0.001, max(abs(v) for row in shown for v in row))
    tokens = s.forward.config.context
    for r, row in enumerate(shown):
        row_index = n - len(shown) + r
        pos = positions[row_index] if positions is not None else row_index
        p.fit_text(
            x,
            y + 31 + r * 17,
            f"{pos}:{s.forward.config.vocab[tokens[pos]]}",
            30,
            7,
            TEXT,
        )
        for col, value in enumerate(row):
            color = mix(
                "#243947",
                GOLD if value >= 0 else "#88b3c3",
                0.15 + 0.85 * abs(value) / maximum,
            )
            p.rect(x + 33 + col * cell, y + 29 + r * 17, cell - 2, 14, color)
    p.line(
        [
            (x + 31, y + 27 + (len(shown) - 1) * 17),
            (x + 31 + d * cell, y + 27 + (len(shown) - 1) * 17),
        ],
        GOLD,
    )
    p.text(x, y + 124, p.tr("matrix_rows", n=len(shown)), 6.5, MUTED, width=width)
