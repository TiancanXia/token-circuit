"""Readable attention cards; all displayed numbers come from stored traces."""

from ..simulation.attention_probe import inspect_key
from .painting import GOLD, TEXT, MUTED, mix
from .attention_glyphs import BLUE


def button(p, x, y, w, label, action, value):
    tags = ("attention-choice", f"probe:{action}:{value}")
    p.rect(x, y, w, 25, "#253e51", tags=tags)
    p.fit_text(x + w / 2, y + 5, label, w - 6, 11, TEXT, anchor="n", tags=tags)


def miniature(p, x, y, values, color, scale):
    width = 80 / max(1, len(values))
    p.line([(x, y + 8), (x + 80, y + 8)], "#44606c")
    for i, value in enumerate(values):
        if value:
            height = abs(value) / scale * 8
            p.rect(
                x + i * width,
                y + 8 - height if value > 0 else y + 8,
                max(1, width - 3),
                max(1, height),
                color,
            )


def instrument(p, s, data, head):
    config = s.forward.config
    row, key = s.attention_probe.indices(data)
    item = inspect_key(data, config, head, row, key)
    trace = data["traces"][head]
    context = config.context
    query = config.vocab[context[item.position]]
    button(p, 48, 185, 24, "‹", "row", -1)
    button(p, 232, 185, 24, "›", "row", 1)
    p.fit_text(
        79, 190, p.tr("probe.query", position=item.position, token=query), 147, 14, GOLD
    )
    button(
        p,
        268,
        185,
        166,
        p.tr("probe.head", head=head + 1, kv=item.kv_head + 1),
        "head",
        1,
    )
    if len(context) > 6:
        button(p, 548, 185, 23, "‹", "page", -1)
        button(p, 577, 185, 23, "›", "page", 1)
    else:
        p.fit_text(450, 192, p.tr("probe.pick"), 150, 11, MUTED)

    start = key // 6 * 6
    indices = list(range(start, min(start + 6, len(context))))
    step = 552 / len(indices)
    center = (154, 212)
    for col, position in enumerate(indices):
        x = 48 + col * step
        probability = trace["weights"][row][position]
        masked, active = position > item.position, position == key
        color = MUTED if masked else GOLD if active else BLUE
        p.line(
            [center, (x + step / 2, 223)],
            mix("#203847", color, 0.2 + probability),
            2 if active else 1,
        )
        if active and not masked:
            t = (s.time * 0.7) % 1
            p.rect(
                center[0] * (1 - t) + (x + step / 2) * t - 2,
                center[1] * (1 - t) + 223 * t - 2,
                4,
                4,
                GOLD,
            )
        tags = ("attention-choice", f"probe:key:{position}")
        p.rect(x, 223, step - 7, 46, "#3d514a" if active else "#1c3342", tags=tags)
        p.rect(x, 223, step - 7, 2, color, tags=tags)
        p.rect(
            x + step - 17,
            228,
            4,
            4,
            BLUE if position < data["reused"] else GOLD,
            tags=tags,
        )
        p.fit_text(
            x + 8,
            228,
            f"{position} · {config.vocab[context[position]]}",
            step - 21,
            14,
            TEXT,
            tags=tags,
        )
        p.fit_text(
            x + 8,
            249,
            p.tr("probe.masked") if masked else f"{probability:.1%}",
            step - 20,
            14,
            color,
            tags=tags,
        )
        p.rect(x + 7, 267, (step - 20) * probability, 2, color, tags=tags)

    status = (
        "probe.masked"
        if item.masked
        else "probe.cached"
        if item.cached
        else "probe.new"
    )
    p.fit_text(
        48,
        274,
        p.tr("probe.key", index=key, score=item.score, status=p.tr(status)),
        552,
        11,
        MUTED,
    )
    scale = max(
        0.001, *(abs(value) for value in item.value + item.contribution + item.output)
    )
    for x, label, values, color in (
        (48, "probe.content", item.value, BLUE),
        (242, "probe.part", item.contribution, GOLD),
        (436, "probe.total", item.output, TEXT),
    ):
        p.rect(x, 294, 164, 36, "#1c3342")
        p.text(x + 10, 295, p.tr(label), 14, color)
        p.text(x + 10, 315, f"{values[0]:+.3f}", 11, TEXT, mono=True)
        miniature(p, x + 78, 310, values, color, scale)
    p.text(222, 308, "×", 14, MUTED)
    p.text(415, 308, "+…", 10, MUTED, mono=True)
