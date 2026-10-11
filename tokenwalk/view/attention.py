"""Attention overview: selected-stage data, head layout and the actual phase."""

from .charts import result_for, matrix, bars
from .attention_glyphs import head_links
from .painting import GOLD, TEXT, MUTED
from .attention_probe import instrument


def phase_caption(p, s, data):
    c = s.forward.config
    phase = p.tr("phase." + data["phase"])
    p.fit_text(
        48,
        306,
        f"{c.attention_mode} / {c.heads}Q · {c.num_kv_heads}KV / " + phase,
        552,
        11,
        GOLD,
    )


def flow_caption(p, s, data):
    traces = data.get("traces")
    needed = (
        any(traces[0]["mask_applied"])
        if traces
        else any(pos < len(s.forward.config.context) - 1 for pos in data["positions"])
    )
    p.fit_text(
        48,
        321,
        p.tr(
            "attention_flow",
            visibility=p.tr("causal_visibility" if needed else "all_visible"),
        ),
        552,
        11,
        TEXT,
    )


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
        # Layout is architecture data, never a replacement for an uncomputed result.
        head_links(p, s)
        title = s.forward.stages[cp.index].title if cp else "Attention"
        p.fit_text(48, 155, p.tr("await_attention_stage", title=title), 552, 12, GOLD)
        phase_caption(p, s, s.forward.attention_metadata())
        p.fit_text(48, 321, p.tr("attention_layout"), 552, 11, TEXT)
        return
    data = shown.details
    kind = shown.stage.kind
    if kind == "attention":
        instrument(p, s, data, s.attention_probe.head % s.forward.config.heads)
        return
    if kind in ("qkv", "rope"):
        for col, key in enumerate(("q", "k", "v")):
            positions = data["positions"] if key == "q" or kind == "qkv" else None
            matrix(
                p,
                48 + col * 188,
                160,
                data[key + "_matrix"],
                key.upper(),
                s,
                positions=positions,
            )
    elif kind == "project":
        p.text(48, 164, "Packed heads / input", 12, GOLD, mono=True)
        p.text(375, 164, "Y = Input W_oᵀ", 12, GOLD, mono=True)
        bars(p, 48, 214, data["input"], 218, 52)
        bars(p, 375, 214, shown.values, 218, 52)
        p.text(307, 228, "→", 22, GOLD)
        p.text(48, 283, f"d_model = {s.forward.config.dim}", 10, MUTED, mono=True)
        p.text(375, 283, "W_o / channel mixing", 10, MUTED, mono=True)
    phase_caption(p, s, data)
    flow_caption(p, s, data)
