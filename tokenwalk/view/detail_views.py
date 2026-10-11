"""Third-level diagrams read detached/stored results; they never run the model."""

from .painting import GOLD, TEXT, MUTED
from .attention_probe import instrument
from .attention_glyphs import BLUE, vector, head_links, packed_heads, mask_grid


def qkv(p, s, node, data, head):
    c = s.forward.config
    if node == 0:
        vector(p, 48, 204, data["input"], "X / RMSNorm output")
        p.text(48, 277, f"d_model = {c.dim}  ·  X: 1×{c.dim}", 11, BLUE, mono=True)
    elif node == 4:
        head_links(p, s, head)
    else:
        key = ("q", "k", "v")[node - 1]
        channel = (head if key == "q" else c.kv_head_for(head)) * c.head_dim
        weights = data["weights"][key]
        vector(
            p,
            48,
            204,
            weights[channel],
            f"W{key} [out {channel}] / {len(weights)}×{c.dim}",
            color=BLUE,
        )
        p.text(
            48,
            278,
            f"{key.upper()}[{channel}] = Σ X[j] W{key}[{channel},j] = {data[key][channel]:+.4f}",
            14,
            GOLD,
            mono=True,
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
            9,
            TEXT,
            anchor="n",
        )
        p.text(x + (step - 6) / 2, 248, "K / V", 9, color, anchor="n", mono=True)
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
        11,
        BLUE,
    )
    size = 2 * length * c.kv_dim
    p.fit_text(
        48,
        288,
        p.tr("cache_size", n=size, saved=2 * length * (c.dim - c.kv_dim)),
        552,
        11,
        MUTED,
    )
    if not s.cache_enabled:
        p.fit_text(48, 304, p.tr("cache_off_note"), 552, 10, MUTED)


def rotate(p, s, node, data, head):
    c = s.forward.config
    pos = data["position"]
    if node == 2:
        cache_cells(p, s, data)
        return
    if node == 0:
        p.text(
            48,
            205,
            f"position = {pos}   ·   d_head = {c.head_dim}",
            15,
            GOLD,
            mono=True,
        )
        p.text(48, 248, "θ_i = position / 10000^(2i / d_head)", 14, BLUE, mono=True)
        p.fit_text(48, 290, p.tr("rope_absolute"), 552, 12, MUTED)
    else:
        start = head * c.head_dim
        ks = c.kv_head_for(head) * c.head_dim
        p.fit_text(48, 181, "Q / K → R(θ) → Q′ / K′", 552, 12, MUTED, mono=True)
        for y, key, offset, name in (
            (197, "q", start, f"Q{head + 1}"),
            (258, "k", ks, f"K{c.kv_head_for(head) + 1}"),
        ):
            vector(
                p,
                48,
                y,
                data["before_" + key][offset : offset + c.head_dim],
                name + " / before",
                width=231,
                color=BLUE,
            )
            vector(
                p,
                360,
                y,
                data[key][offset : offset + c.head_dim],
                "after RoPE",
                width=231,
            )
            p.text(308, y + 26, "→", 22, GOLD)


def attend(p, s, node, data, head):
    c = s.forward.config
    trace = data["traces"][head]
    if node == 0:
        cache_cells(p, s, data)
        return
    if node == 3:
        instrument(p, s, data, head)
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
            vector(p, 48, 209, scores[-9:], "scores unchanged / all keys visible")
            p.fit_text(48, 281, p.tr("mask_bypass"), 552, 14, TEXT)
        return
    weights = trace["weights"][-1]
    start = max(0, len(weights) - 9)
    values = trace["scores"][-1] if node == 1 else weights
    label = (
        "Q Kᵀ / √d_head"
        if node == 1
        else "softmax(scores)"
        if node == 3
        else "p (all keys)"
    )
    vector(p, 48, 202, values[start:], label)
    p.fit_text(
        48,
        181,
        p.tr(
            "attention_positions",
            start=start,
            end=len(weights) - 1,
            pos=data["positions"][-1],
        ),
        552,
        11,
        MUTED,
    )
    if node == 1:
        qs = head * c.head_dim
        ks = trace["kv_head"] * c.head_dim
        q = data["q_matrix"][-1][qs : qs + c.head_dim]
        k = data["k_matrix"][-1][ks : ks + c.head_dim]
        terms = " + ".join(f"{a:+.3f}×{b:+.3f}" for a, b in zip(q, k))
        p.fit_text(
            48,
            278,
            f"last key: ({terms}) / √{c.head_dim} = {values[-1]:+.4f}",
            552,
            11,
            GOLD,
            mono=True,
        )
    elif node == 3:
        p.text(48, 278, f"Σ p = {sum(weights):.6f}", 16, GOLD, mono=True)
    else:
        output = trace["outputs"][-1]
        ks = trace["kv_head"] * c.head_dim
        if start:
            earlier = sum(
                w * v[ks] for w, v in zip(weights[:start], data["v_matrix"][:start])
            )
            shown = sum(
                w * v[ks] for w, v in zip(weights[start:], data["v_matrix"][start:])
            )
            formula = p.tr(
                "weighted_crop", shown=shown, earlier=earlier, total=output[0]
            )
        else:
            formula = (
                " + ".join(
                    f"{w:.2f}×{v[ks]:+.2f}" for w, v in zip(weights, data["v_matrix"])
                )
                + f" = {output[0]:+.4f}"
            )
        p.fit_text(48, 270, formula, 552, 10, BLUE, mono=True)
        p.text(
            48,
            292,
            f"HEAD {head + 1} → [{', '.join(f'{x:+.4f}' for x in output)}]",
            11,
            GOLD,
            mono=True,
        )


def project(p, s, node, data, head):
    if node == 0:
        packed_heads(p, s, data["input"], head, caption="packed_input")
    elif node == 1:
        row = head * s.forward.config.head_dim
        vector(p, 48, 206, data["weights"][row], f"W_o row {row}", color=BLUE)
        p.text(
            48,
            282,
            f"Y[{row}] = Σ Input[j] W_o[{row},j] = {s.rooms.active.result.values[row]:+.4f}",
            14,
            GOLD,
            mono=True,
        )
    else:
        vector(p, 48, 206, s.rooms.active.result.values, "Y / attention branch output")
        p.fit_text(48, 282, p.tr("project_residual"), 552, 10, BLUE)


DRAWERS = {"qkv": qkv, "rope": rotate, "attention": attend, "project": project}


def draw_detail(p, s):
    visit = s.rooms.active
    cp = s.near_checkpoint or s.next_checkpoint or visit.spec.checkpoints()[-1]
    node = cp.index
    p.fit_text(48, 153, visit.spec.nodes[node], 370, 18, GOLD, mono=True)
    c = s.forward.config
    kv = c.kv_head_for(visit.head)
    p.text(
        600,
        160,
        f"N / Q{visit.head + 1} → KV{kv + 1}",
        11,
        BLUE,
        anchor="ne",
        mono=True,
    )
    if node > visit.progress:
        p.text(48, 218, p.tr("detail_order"), 12, MUTED)
    else:
        DRAWERS[visit.spec.kind](p, s, node, visit.result.details, visit.head)
    if visit.spec.kind == "attention" and node == 3 and node <= visit.progress:
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
        320,
        p.tr(key),
        552,
        13,
        TEXT,
        tags="detail-explanation",
    )
