"""One bilingual Canvas panel for choosing any actual block/stage."""

from .painting import TEXT, MUTED, GOLD


_LABELS = {
    "norm_attn": "RMSNorm · Attn",
    "norm_ffn": "RMSNorm · FFN",
    "residual_attn": "Residual · Attn",
    "residual_ffn": "Residual · FFN",
}


def navigation_panel(p, s, picker):
    p.rect(86, 131, 728, 390, "#091722")
    p.line([(86, 521), (86, 131), (814, 131), (814, 521)], "#577682")
    p.text(110, 148, p.tr("navigation_title"), 16, TEXT)
    p.text(110, 176, p.tr("navigation_note"), 8, MUTED, width=680)
    for slot, column in enumerate(picker.visible_columns):
        indices = picker.columns[column]
        block = s.forward.stages[indices[0]].block
        style = (
            "Dense FFN"
            if block >= 0 and s.forward.config.blocks[block] == "dense"
            else "MoE FFN"
        )
        title = p.tr("global_stages") if block < 0 else f"Block {block + 1} · {style}"
        x = 110 + slot * 234
        p.text(x, 200, title, 11, GOLD)
        for row, index in enumerate(indices):
            y = 226 + row * 22
            selected = index == picker.selected
            tags = ("navigation-select", f"stage:{index}")
            p.rect(
                x,
                y,
                216,
                20,
                "#46574e" if selected else "#182d3b",
                tags=tags + ("navigation-row",),
            )
            stage = s.forward.stages[index]
            label = f"{index + 1:02d}  " + _LABELS.get(stage.kind, stage.title)
            p.fit_text(
                x + 9,
                y + 4,
                label,
                198,
                10,
                GOLD if selected else TEXT,
                mono=True,
                tags=tags,
            )
    p.text(110, 491, p.tr("navigation_keys"), 8, MUTED, width=680)
