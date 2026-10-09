"""Attention overview: selected-stage data, head layout and the actual phase."""

from .charts import result_for, matrix, bars
from .attention_glyphs import head_links
from .painting import GOLD, TEXT, MUTED, mix


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
    else:
        for head, weights in enumerate(data["all_attention"][:2]):
            x = 58 + head * 283
            n = len(weights[0])
            keys = list(range(max(0, n - 6), n))
            rows = list(zip(data["positions"], weights))[-6:]
            cell = 15
            kv = data["traces"][head]["kv_head"]
            p.text(x, 157, f"Q HEAD {head + 1} → KV {kv + 1}", 10, GOLD, mono=True)
            for col, key in enumerate(keys):
                p.text(
                    x + 40 + col * cell + 6,
                    180,
                    str(key),
                    7,
                    MUTED,
                    anchor="n",
                    mono=True,
                )
            for row_index, (position, row) in enumerate(rows):
                label = f"{position}:{s.forward.config.vocab[s.forward.config.context[position]]}"
                p.fit_text(x, 202 + row_index * cell, label, 37, 7, TEXT)
                for col, key in enumerate(keys):
                    px, py = x + 40 + col * cell, 198 + row_index * cell
                    p.rect(
                        px,
                        py,
                        cell - 2,
                        cell - 2,
                        mix("#263d49", GOLD, min(1, row[key] * 5)),
                    )
                    if key > position:
                        p.line([(px + 2, py + 2), (px + 10, py + 10)], "#78909b")
                    if position == n - 1:
                        p.line(
                            [(px, py + cell - 1), (px + cell - 2, py + cell - 1)], GOLD
                        )
            p.fit_text(
                x,
                286,
                p.tr("attention_crop", rows=len(rows), keys=len(keys)),
                250,
                9,
                MUTED,
            )
    phase_caption(p, s, data)
    flow_caption(p, s, data)
