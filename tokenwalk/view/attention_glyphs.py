"""Shared attention layout, packing and visibility diagrams; no model execution."""

import math
from .painting import GOLD, TEXT, MUTED
from .charts import bars
from .head_controls import head_tags

BLUE = "#87b8cc"


def vector(p, x, y, values, label, width=552, color=GOLD, height=32, **kw):
    p.fit_text(x, y, label, width, 16, color, **kw)
    bars(p, x, y + 28, values, width, height, **kw)


def head_links(p, s, head=0):
    config = s.forward.config
    heads, kv = config.heads, config.num_kv_heads
    start = head // 8 * 8
    shown_heads = range(start, min(start + 8, heads))
    count = len(shown_heads)
    shown_kv = sorted({config.kv_head_for(h) for h in shown_heads})
    selected = config.kv_head_for(head)

    def kv_x(group):
        return (
            324
            if len(shown_kv) == 1
            else 80 + shown_kv.index(group) * 500 / (len(shown_kv) - 1)
        )

    for slot, h in enumerate(shown_heads):
        x = 324 if count == 1 else 70 + slot * 520 / (count - 1)
        color = GOLD if h == head else "#476575"
        p.line(
            [(x, 225), (kv_x(config.kv_head_for(h)), 280)], color, 2 if h == head else 1
        )
        tags = head_tags(h)
        p.oval(x - 15, 209, 30, 30, "#375365" if h == head else "#203949", tags=tags)
        p.text(x, 216, f"Q{h + 1}", 10, color, anchor="n", mono=True, tags=tags)
    for k in shown_kv:
        x = kv_x(k)
        tags = head_tags(
            head
            if k == selected
            else next(h for h in shown_heads if config.kv_head_for(h) == k)
        )
        p.rect(x - 26, 272, 52, 27, "#35565b", tags=tags)
        p.text(
            x,
            278,
            f"KV{k + 1}",
            9,
            GOLD if k == selected else BLUE,
            anchor="n",
            mono=True,
            tags=tags,
        )
    p.fit_text(
        48,
        187,
        p.tr(
            "head_groups",
            mode=config.attention_mode,
            q=heads,
            kv=kv,
            start=start + 1,
            end=start + count,
        ),
        552,
        15,
        TEXT,
    )


def packed_heads(p, s, values, head):
    c = s.forward.config
    start_head = head // 8 * 8
    count = min(c.heads - start_head, 8)
    width = 552 / count
    for slot, h in enumerate(range(start_head, start_head + count)):
        start = h * c.head_dim
        vector(
            p,
            48 + slot * width,
            207,
            values[start : start + c.head_dim],
            f"HEAD {h + 1}",
            width - 5,
            GOLD if h == head else BLUE,
            tags=head_tags(h),
        )
    if count < c.heads:
        p.fit_text(
            48,
            285,
            p.tr(
                "concat_shape",
                q=c.heads,
                d=c.head_dim,
                total=c.dim,
                start=start_head + 1,
                end=start_head + count,
            ),
            552,
            12,
            MUTED,
        )


def projected_output(p, s, result, head):
    vector(
        p,
        48,
        188,
        result.details["head_contributions"][head],
        p.tr("head_contribution", n=head + 1),
        height=24,
        tags="head-values",
    )
    vector(
        p,
        48,
        252,
        result.values,
        p.tr("all_heads_output"),
        color=BLUE,
        height=24,
        tags="attention-total-output",
    )


def mask_grid(p, data, head):
    """Show the mask actually used for all stored queries, with correct positions."""
    trace = data["traces"][head]
    n = len(trace["masked"][0])
    keys = list(range(max(0, n - 9), n))
    queries = list(zip(data["positions"], trace["masked"]))[-6:]
    cell = 15
    x, y = 71, 210
    for column, key in enumerate(keys):
        p.text(x + column * cell + 6, y - 17, str(key), 7, MUTED, anchor="n", mono=True)
    for r, (position, row) in enumerate(queries):
        p.text(x - 21, y + r * cell + 2, str(position), 7, TEXT, mono=True)
        for column, key in enumerate(keys):
            px, py = x + column * cell, y + r * cell
            hidden = row[key] == -math.inf
            p.rect(
                px,
                py,
                cell - 2,
                cell - 2,
                "#223642" if hidden else "#788a71",
                tags="mask-cell",
            )
            if hidden:
                # A semantic marker supports both exported previews and Tk checks.
                p.rect(px + 5, py + 5, 3, 3, "#849cac", tags="mask-hidden")
                p.line([(px + 2, py + 2), (px + 10, py + 10)], BLUE)
    p.fit_text(
        245, 217, p.tr("mask_axes", rows=len(queries), keys=len(keys)), 355, 13, TEXT
    )
    p.fit_text(245, 257, p.tr("mask_last"), 355, 15, BLUE)
