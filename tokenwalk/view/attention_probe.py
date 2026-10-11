"""Token connections and scores/weights from a stored attention trace."""

from ..simulation.attention_probe import inspect_key
from .painting import GOLD, TEXT, MUTED, mix
from .attention_glyphs import BLUE


def button(p, x, y, w, label, action, value):
    tags = ("attention-choice", f"probe:{action}:{value}")
    p.rect(x, y, w, 25, "#253e51", tags=tags)
    p.fit_text(x + w / 2, y + 5, label, w - 6, 12, TEXT, anchor="n", tags=tags)


def instrument(p, s, data, head, scores=False):
    config = s.forward.config
    row, key = s.attention_probe.indices(data)
    item = inspect_key(data, config, head, row, key)
    trace = data["traces"][head]
    context = config.context
    query = config.vocab[context[item.position]]
    if len(data["positions"]) > 1:
        button(p, 48, 185, 24, "‹", "row", -1)
        button(p, 232, 185, 24, "›", "row", 1)
    p.fit_text(
        79, 188, p.tr("probe.query", position=item.position, token=query), 147, 17, GOLD
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
        p.fit_text(450, 191, p.tr("probe.pick"), 150, 12, MUTED)

    start = key // 6 * 6
    indices = list(range(start, min(start + 6, len(context))))
    step = 552 / len(indices)
    center = (154, 212)
    for col, position in enumerate(indices):
        x = 48 + col * step
        probability = trace["weights"][row][position]
        masked, active = not scores and position > item.position, position == key
        color = MUTED if masked else GOLD if active else BLUE
        p.line(
            [center, (x + step / 2, 238)],
            mix("#203847", color, 0.6 if scores else 0.3 + probability),
            2 if active else 1 if scores else 1 + 3 * probability,
        )
        if active and not masked:
            t = (s.time * 0.7) % 1
            p.rect(
                center[0] * (1 - t) + (x + step / 2) * t - 2,
                center[1] * (1 - t) + 238 * t - 2,
                4,
                4,
                GOLD,
            )
        tags = ("attention-choice", f"probe:key:{position}")
        p.rect(x, 238, step - 7, 58, "#3d514a" if active else "#1c3342", tags=tags)
        p.rect(x, 238, step - 7, 2, color, tags=tags)
        p.fit_text(
            x + 9, 244, config.vocab[context[position]], step - 21, 19, TEXT, tags=tags
        )
        value = (
            f"{trace['scores'][row][position]:+.2f}"
            if scores
            else p.tr("probe.masked")
            if masked
            else f"{probability:.1%}"
        )
        p.fit_text(x + 9, 272, value, step - 20, 16, color, tags=tags)
        if not scores:
            p.rect(x + 8, 294, (step - 23) * probability, 2, color, tags=tags)

    if len(context) > 6:
        if scores:
            caption = p.tr(
                "probe.range", start=start + 1, end=indices[-1] + 1, total=len(context)
            )
        else:
            omitted = 1 - sum(trace["weights"][row][i] for i in indices)
            caption = p.tr(
                "probe.crop",
                shown=len(indices),
                total=len(context),
                rest=max(0, omitted),
            )
    else:
        caption = p.tr("probe.scores_note" if scores else "probe.weights_note")
    p.fit_text(48, 310, caption, 552, 15, TEXT, tags="detail-explanation")
