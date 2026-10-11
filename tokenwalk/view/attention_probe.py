"""Interactive attention instrument; draws existing traces without computation."""

from ..simulation.attention_probe import inspect_key
from .painting import GOLD, TEXT, MUTED, mix
from .attention_glyphs import BLUE


def button(p, x, y, w, label, action, value):
    tags = ("attention-choice", f"probe:{action}:{value}")
    p.rect(x, y, w, 21, "#253e51", tags=tags)
    p.fit_text(x + w / 2, y + 4, label, w - 6, 10, GOLD, anchor="n", tags=tags)


def miniature(p, x, y, values, color, scale):
    width = 132 / max(1, len(values))
    p.line([(x, y + 10), (x + 132, y + 10)], "#3b5361")
    for i, v in enumerate(values):
        if v == 0:
            continue
        height = abs(v) / scale * 9
        p.rect(
            x + i * width,
            y + 10 - height if v >= 0 else y + 10,
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
    button(p, 48, 179, 24, "‹", "row", -1)
    button(p, 232, 179, 24, "›", "row", 1)
    p.fit_text(
        79, 183, p.tr("probe.query", position=item.position, token=query), 147, 12, GOLD
    )
    button(
        p,
        268,
        179,
        154,
        f"{config.attention_mode} · Q{head + 1} → KV{item.kv_head + 1}",
        "head",
        1,
    )
    p.fit_text(
        438, 184, p.tr("probe.pick"), 102 if len(context) > 6 else 162, 10, MUTED
    )
    start = key // 6 * 6
    indices = list(range(start, min(start + 6, len(context))))
    step = 552 / len(indices)
    center = (152, 202)
    for col, j in enumerate(indices):
        x = 48 + col * step
        probability = trace["weights"][row][j]
        masked = j > item.position
        active = j == key
        color = MUTED if masked else GOLD if active else BLUE
        p.line(
            [center, (x + step / 2, 215)],
            mix("#203847", color, min(1, 0.2 + probability)),
            2 if active else 1,
        )
        if active and not masked:
            t = (s.time * 0.7) % 1
            px = center[0] * (1 - t) + (x + step / 2) * t
            py = center[1] * (1 - t) + 215 * t
            p.rect(px - 2, py - 2, 4, 4, GOLD)
        tags = ("attention-choice", f"probe:key:{j}")
        p.rect(x, 215, step - 7, 45, "#394d49" if active else "#1c3342", tags=tags)
        p.rect(x, 215, 3, 45, color, tags=tags)
        p.rect(
            x + step - 17, 219, 4, 4, BLUE if j < data["reused"] else GOLD, tags=tags
        )
        p.fit_text(
            x + 8,
            220,
            f"{j} · {config.vocab[context[j]]}",
            step - 19,
            12,
            color,
            tags=tags,
        )
        label = p.tr("probe.masked") if masked else f"{probability:.1%}"
        p.text(x + 8, 240, label, 10, color, tags=tags)
        p.rect(x + 7, 258, (step - 20) * probability, 2, color, tags=tags)
    if len(context) > 6:
        button(p, 548, 179, 23, "‹", "page", -1)
        button(p, 577, 179, 23, "›", "page", 1)
    p.fit_text(
        48,
        269,
        p.tr(
            "probe.key",
            index=key,
            score=item.score,
            status=p.tr(
                "probe.masked"
                if item.masked
                else "probe.cached"
                if item.cached
                else "probe.new"
            ),
        ),
        552,
        10,
        GOLD,
    )
    scale = max(0.001, *(abs(v) for v in item.value + item.contribution + item.output))
    for x, label, values, color in (
        (48, "V", item.value, BLUE),
        (242, "p × V", item.contribution, GOLD),
        (436, "Σ pV", item.output, TEXT),
    ):
        p.text(x, 290, label, 10, color, mono=True)
        miniature(p, x + 31, 286, values, color, scale)
    p.fit_text(
        48,
        310,
        p.tr("probe.contribution", value=item.contribution[0], total=item.output[0]),
        552,
        10,
        TEXT,
    )
