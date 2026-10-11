"""Third-level diagrams read detached/stored results; they never run the model."""

from .painting import GOLD, TEXT, MUTED
from .attention_probe import instrument
from .attention_glyphs import (
    BLUE,
    vector,
    head_links,
    packed_heads,
    mask_grid,
    projected_output,
)
from .head_controls import head_selector
from .charts import bars


def qkv(p, s, node, data, head):
    c = s.forward.config
    if node == 0:
        vector(p, 48, 204, data["input"], "Input X")
    elif node == 4:
        head_links(p, s, head)
    else:
        key = ("q", "k", "v")[node - 1]
        channel = (head if key == "q" else c.kv_head_for(head)) * c.head_dim
        vector(p, 48, 204, data["input"], "Input X", width=231, color=BLUE)
        p.text(300, 200, "W_" + key.upper(), 17, GOLD)
        p.text(308, 232, "→", 22, GOLD)
        vector(
            p,
            360,
            204,
            data[key][channel : channel + c.head_dim],
            f"{key.upper()}{(head if key == 'q' else c.kv_head_for(head)) + 1}",
            width=231,
            tags="head-values",
        )


def cache_cells(p, s, data, head):
    c = s.forward.config
    reused = data["reused"]
    length = len(c.context)
    start = max(0, length - 9)
    step = 552 / (length - start)
    offset = c.kv_head_for(head) * c.head_dim
    for i in range(start, length):
        x = 48 + (i - start) * step
        old = i < reused
        color = BLUE if old else GOLD
        p.rect(x, 218, step - 6, 59, "#233d4b" if old else "#495346")
        p.fit_text(
            x + (step - 6) / 2,
            225,
            f"{i}: {c.vocab[c.context[i]]}",
            step - 6,
            15,
            TEXT,
            anchor="n",
        )
        for y, key in ((251, "k"), (265, "v")):
            p.text(x + 5, y - 2, key.upper(), 9, color)
            bars(
                p,
                x + 19,
                y,
                data[key + "_matrix"][i][offset : offset + c.head_dim],
                step - 28,
                10,
                tags="head-values",
            )
    p.fit_text(
        48,
        187,
        p.tr(
            "cache_stats",
            phase=p.tr("phase." + data["phase"]),
            old=reused,
            new=length - reused,
            shown=length - start,
        ),
        552,
        15,
        BLUE,
    )
    p.fit_text(48, 289, p.tr("cache_legend"), 552, 14, MUTED)


def rotate(p, s, node, data, head):
    c = s.forward.config
    pos = data["position"]
    if node == 2:
        cache_cells(p, s, data, head)
        return
    if node == 0:
        p.text(48, 216, f"Position · {pos}", 24, GOLD)
        p.fit_text(48, 270, p.tr("brief.rope.0"), 552, 17, TEXT)
    else:
        start = head * c.head_dim
        ks = c.kv_head_for(head) * c.head_dim
        for y, key, offset, name in (
            (188, "q", start, f"Q{head + 1}"),
            (245, "k", ks, f"K{c.kv_head_for(head) + 1}"),
        ):
            vector(
                p,
                48,
                y,
                data["before_" + key][offset : offset + c.head_dim],
                name + " / " + p.tr("before"),
                width=231,
                color=BLUE,
                tags="head-values",
            )
            vector(
                p,
                360,
                y,
                data[key][offset : offset + c.head_dim],
                "RoPE / " + p.tr("after"),
                width=231,
                tags="head-values",
            )
            p.text(308, y + 26, "→", 22, GOLD)


def attend(p, s, node, data, head):
    trace = data["traces"][head]
    if node == 0:
        cache_cells(p, s, data, head)
        return
    if node in (1, 3):
        instrument(p, s, data, head, scores=node == 1)
        return
    if node == 5:
        packed_heads(p, s, s.rooms.active.result.values, head)
        return
    if node == 2:
        needed = any(trace["mask_applied"])
        if needed:
            mask_grid(p, data, head)
        else:
            scores = trace["scores"][-1]
            positions = p.tr(
                "attention_positions",
                start=max(0, len(scores) - 9),
                end=len(scores) - 1,
                pos=data["positions"][-1],
            )
            p.fit_text(
                48,
                187,
                p.tr("phase." + data["phase"]) + " · " + positions,
                552,
                11,
                BLUE,
            )
            vector(p, 48, 209, scores[-9:], p.tr("probe.match"))
            p.fit_text(48, 281, p.tr("mask_bypass"), 552, 14, TEXT)
        return
    vector(
        p, 48, 206, trace["outputs"][-1], f"Head {head + 1} output", tags="head-values"
    )


def project(p, s, node, data, head):
    if node == 0:
        packed_heads(p, s, data["input"], head)
    elif node == 1:
        c = s.forward.config
        start = head * c.head_dim
        vector(
            p,
            48,
            206,
            data["input"][start : start + c.head_dim],
            f"Head {head + 1} input",
            width=231,
            color=BLUE,
        )
        p.fit_text(
            295, 200, f"W_o[:, {start}:{start + c.head_dim}]", 92, 12, GOLD, anchor="n"
        )
        p.text(308, 232, "→", 22, GOLD)
        vector(
            p,
            360,
            206,
            data["head_contributions"][head],
            p.tr("head_contribution", n=head + 1),
            width=231,
            tags="head-values",
        )
    else:
        projected_output(p, s, s.rooms.active.result, head)


DRAWERS = {"qkv": qkv, "rope": rotate, "attention": attend, "project": project}


def draw_detail(p, s):
    visit = s.rooms.active
    cp = s.near_checkpoint or s.next_checkpoint or visit.spec.checkpoints()[-1]
    node = visit.spec.operation_index(cp.index)
    p.fit_text(48, 151, s.checkpoint_title(cp.index), 280, 22, GOLD)
    head_selector(p, s, visit.head)
    if cp.index > visit.progress:
        p.text(48, 218, p.tr("detail_order"), 12, MUTED)
    else:
        DRAWERS[visit.spec.kind](p, s, node, visit.result.details, visit.head)
    if visit.spec.kind == "attention" and node in (1, 3) and cp.index <= visit.progress:
        return
    key = f"brief.{visit.spec.kind}.{node}"
    if visit.spec.kind == "attention" and node == 0:
        key = "brief.attention.read." + visit.result.details["phase"]
    elif (
        visit.spec.kind == "attention"
        and node == 2
        and not any(visit.result.details["traces"][0]["mask_applied"])
    ):
        key = "brief.attention.visible"
    elif visit.spec.kind == "rope" and node == 2:
        key = "brief.rope.ready." + visit.result.details["phase"]
    p.fit_text(
        48,
        310,
        p.tr(key),
        552,
        17,
        TEXT,
        tags="detail-explanation",
    )
