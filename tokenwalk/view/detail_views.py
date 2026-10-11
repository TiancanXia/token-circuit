"""Third-level diagrams read detached/stored results; they never run the model."""

from .painting import GOLD, TEXT, MUTED
from .attention_probe import instrument
from .attention_glyphs import BLUE, vector, head_links, packed_heads, mask_grid


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
            key.upper(),
            width=231,
        )


def cache_cells(p, s, data):
    c = s.forward.config
    reused = data["reused"]
    length = len(c.context)
    start = max(0, length - 9)
    step = 552 / (length - start)
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
        p.text(x + (step - 6) / 2, 253, "K / V", 12, color, anchor="n")
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
        cache_cells(p, s, data)
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
            )
            vector(
                p,
                360,
                y,
                data[key][offset : offset + c.head_dim],
                "RoPE / " + p.tr("after"),
                width=231,
            )
            p.text(308, y + 26, "→", 22, GOLD)


def attend(p, s, node, data, head):
    trace = data["traces"][head]
    if node == 0:
        cache_cells(p, s, data)
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
    vector(p, 48, 206, trace["outputs"][-1], f"Head {head + 1} output")


def project(p, s, node, data, head):
    if node == 0:
        packed_heads(p, s, data["input"], head)
    elif node == 1:
        vector(p, 48, 206, data["input"], "Input", width=231, color=BLUE)
        p.text(300, 200, "W_o", 17, GOLD)
        p.text(308, 232, "→", 22, GOLD)
        vector(p, 360, 206, s.rooms.active.result.values, "Output", width=231)
    else:
        vector(p, 48, 206, s.rooms.active.result.values, "Output")


DRAWERS = {"qkv": qkv, "rope": rotate, "attention": attend, "project": project}


def draw_detail(p, s):
    visit = s.rooms.active
    cp = s.near_checkpoint or s.next_checkpoint or visit.spec.checkpoints()[-1]
    node = cp.index
    p.fit_text(48, 151, s.checkpoint_title(node), 400, 22, GOLD)
    c = s.forward.config
    kv = c.kv_head_for(visit.head)
    p.text(
        600,
        160,
        p.tr("probe.head_hint", head=visit.head + 1, count=c.heads)
        if visit.spec.kind == "attention"
        else f"N / Q{visit.head + 1} → KV{kv + 1}",
        11,
        BLUE,
        anchor="ne",
        mono=visit.spec.kind != "attention",
        tags="head-indicator",
    )
    if node > visit.progress:
        p.text(48, 218, p.tr("detail_order"), 12, MUTED)
    else:
        DRAWERS[visit.spec.kind](p, s, node, visit.result.details, visit.head)
    if visit.spec.kind == "attention" and node in (1, 3) and node <= visit.progress:
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
