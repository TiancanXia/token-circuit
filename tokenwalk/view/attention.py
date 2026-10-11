"""Attention overview: selected-stage data, head layout and the actual phase."""

from .charts import result_for
from .attention_glyphs import vector
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
    if shown is None:
        title = s.forward.stages[cp.index].title if cp else "Attention"
        p.fit_text(48, 185, p.tr("await_attention_stage", title=title), 552, 18, GOLD)
        p.fit_text(48, 265, p.tr("attention.walk"), 552, 17, TEXT)
        return
    data = shown.details
    kind = shown.stage.kind
    if kind == "attention":
        p.text(48, 151, "Attention weights", 22, GOLD)
        instrument(p, s, data, s.attention_probe.head % s.forward.config.heads)
        return
    if kind in ("qkv", "rope"):
        for col, key in enumerate(("q", "k", "v")):
            vector(
                p,
                48 + col * 188,
                160,
                data[key + "_matrix"][-1],
                key.upper(),
                width=166,
            )
    elif kind == "project":
        vector(p, 48, 185, data["input"], "Input", width=218)
        vector(p, 375, 185, shown.values, "Output", width=218)
        p.text(307, 228, "→", 22, GOLD)
        p.text(300, 192, "W_o", 16, GOLD)
    p.fit_text(
        48, 310, p.tr("attention." + ("qkv" if kind == "qkv" else kind)), 552, 16, TEXT
    )
