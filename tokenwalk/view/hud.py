"""Compact sidebar and token rail, separate from scene composition."""

from .painting import TEXT, MUTED, GOLD
from .charts import bars
from .context import focused_stage


def sidebar(p, s):
    stage, result = focused_stage(s)
    f = s.forward
    p.line([(646, 142), (646, 516)], "#314651")
    p.fit_text(662, 147, "TOKEN / " + s.token, 214, 15, GOLD)
    p.text(
        662,
        175,
        f"{len(f.config.context)} token · {f.next_index}/{len(f.stages)}",
        11,
        MUTED,
    )
    title = stage.title
    if s.guided and stage.kind == "sample":
        title = "Next token"
    if s.scene == "macro" and s.near_house:
        title = s.near_house.title
        if s.guided and s.near_house.key == "output":
            title = "Output"
    p.fit_text(662, 196, title, 214, 16, TEXT, mono=True)
    p.fit_text(662, 225, p.tr("stage." + stage.kind), 214, 11, GOLD)
    if s.scene == "detail":
        phase = result.details.get("phase")
        status = (
            p.tr("phase." + phase)
            if phase
            else p.tr("wait" if s.rooms.active.index == f.next_index else "processed")
        )
        p.fit_text(662, 245, status, 214, 10, MUTED)
    caption = p.tr(
        "fast_forward"
        if s.scene in ("macro", "micro", "detail")
        else "flying"
        if s.scene == "flight"
        else "transferring"
        if s.scene == "birth"
        else "finished"
    )
    p.rect(662, 267, 214, 32, "#253e51", tags="fast-forward")
    p.text(676, 275, caption, 12, GOLD, tags="fast-forward")
    output_stage = stage.kind in ("unembedding", "softmax", "sample")
    values = result.values if result else []
    label = p.tr("result." + result.label) if result else p.tr("wait")
    in_attention = s.scene in ("micro", "detail") and s.group.house == "attention"
    if in_attention:
        p.text(662, 326, p.tr("attention.walk"), 16, TEXT, width=213)
        p.text(662, 382, p.tr("attention.read"), 13, MUTED, width=213)
    else:
        p.text(662, 313, label, 10, MUTED, width=213)
        bars(p, 662, 350, values, 212, 25)
        p.text(
            662,
            386,
            p.tr("vocab_size", n=len(f.config.vocab))
            if output_stage
            else f"{len(values)} channels"
            if result
            else p.tr("not_computed"),
            10,
            MUTED,
        )
    if result and "selected" in result.details:
        selected = result.details["selected"]
        weights = result.details["weights"]
        p.fit_text(
            662,
            424,
            ", ".join(f"E{i + 1} {w:.0%}" for i, w in zip(selected, weights)),
            214,
            11,
            GOLD,
        )
    elif result and "sampled" in result.details:
        value = s.pending_token if s.guided else result.details["sampled"]
        p.fit_text(
            662,
            424,
            p.tr("complete")
            if s.scene == "done"
            else p.tr("next_token", token=value or ""),
            214,
            11,
            GOLD,
        )
    elif not in_attention:
        p.text(662, 424, p.tr("signs"), 10, MUTED)
    p.rect(662, 452, 214, 28, "#1c3342", tags="model-guide")
    p.text(676, 458, p.tr("guide.button"), 12, GOLD, tags="model-guide")
    button = p.tr(
        "enter"
        if s.scene == "macro"
        else "process_exit"
        if s.scene in ("micro", "detail")
        else "restart"
        if s.scene == "done"
        else "flying"
        if s.scene == "flight"
        else "returning"
    )
    p.rect(662, 488, 214, 34, "#344c48", tags="interact")
    p.text(676, 497, button, 12, TEXT, tags="interact")
    p.text(662, 530, p.tr("back_layers"), 10, MUTED)


def sentence_rail(p, s):
    cells = s.sentence_cells
    count = len(cells)
    step = min(84.0 if s.language == "en" else 32.0, 852 / count)
    width = step - 4
    start = 24 + (852 - step * count) / 2
    current = s.generation if s.guided else count - 1
    for i, ch in enumerate(cells):
        x = start + i * step
        active = i == current
        color = "#b6aa76" if active else "#233b48" if ch is not None else "#080f17"
        p.rect(x, 549, width, 27, color, tags="sentence-cell")
        if ch is not None:
            p.fit_text(
                x + width / 2,
                552,
                ch,
                width,
                15,
                TEXT if active else "#8ea6ad",
                anchor="n",
                tags="sentence-character",
            )
        if active:
            p.line([(x, 581), (x + width, 581)], GOLD, 2)
