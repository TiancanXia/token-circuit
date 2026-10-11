"""Attention overview: selected-stage data, head layout and the actual phase."""

from .charts import result_for
from .attention_glyphs import vector, projected_output
from .head_controls import head_selector
from .painting import GOLD, TEXT
from .attention_probe import instrument


def attention_world(p, s):
    computed = [
        r
        for r in s.forward.results
        if r.stage.block == s.group.block
        and r.stage.kind in ("qkv", "rope", "attention", "project")
    ]
    cp = s.near_checkpoint or s.next_checkpoint
    shown = result_for(s, cp.index) if cp else computed[-1] if computed else None
    c = s.forward.config
    head = s.selected_head
    head_selector(p, s, head)
    if shown is None:
        title = s.forward.stages[cp.index].title if cp else "Attention"
        p.fit_text(48, 185, p.tr("await_attention_stage", title=title), 552, 18, GOLD)
        p.fit_text(48, 265, p.tr("attention.walk"), 552, 17, TEXT)
        return
    data = shown.details
    kind = shown.stage.kind
    if kind == "attention":
        p.fit_text(48, 151, "Attention weights", 280, 22, GOLD)
        instrument(p, s, data, head)
        return
    if kind in ("qkv", "rope"):
        for col, key in enumerate(("q", "k", "v")):
            selected = head if key == "q" else c.kv_head_for(head)
            start = selected * c.head_dim
            vector(
                p,
                48 + col * 188,
                195,
                data[key + "_matrix"][-1][start : start + c.head_dim],
                f"{key.upper()}{selected + 1}",
                width=166,
                tags="head-values",
            )
    elif kind == "project":
        p.text(48, 151, "Output · W_o", 22, GOLD)
        projected_output(p, s, shown, head)
    p.fit_text(
        48, 310, p.tr("attention." + ("qkv" if kind == "qkv" else kind)), 552, 16, TEXT
    )
