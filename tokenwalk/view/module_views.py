"""Detailed diagrams inside module houses, using stored actual forward results."""

from .glyphs import character
from .painting import GOLD, TEXT, MUTED
from .output import output_world
from .charts import result_for, bars
from .attention import attention_world


def embedding_world(p, s):
    tokens = s.forward.config.context[-9:]
    p.text(49, 164, "Embedding / token → vector", 15, GOLD)
    for i, idx in enumerate(tokens):
        x = 67 + i * 56
        character(p, x, 226, ghost=i < len(tokens) - 1)
        p.fit_text(x, 192, s.forward.config.vocab[idx], 52, 10, TEXT, anchor="n")
    r = result_for(s, s.group.indices[0])
    p.text(49, 255, p.tr("prefix", n=len(tokens)), 10, MUTED, width=551)
    if r:
        bars(p, 80, 280, r.values, 470, 27)
        p.text(
            80,
            319,
            p.tr("features", id=s.forward.config.context[-1], n=s.forward.config.dim),
            11,
            GOLD,
        )


def ffn_world(p, s):
    config = s.forward.config
    if config.blocks[s.group.block] == "dense":
        for col, index in enumerate(s.group.indices):
            x = 48 + col * 189
            stage = s.forward.stages[index]
            r = result_for(s, index)
            p.text(x, 167, stage.title, 12, GOLD if r else MUTED, mono=True)
            bars(p, x, 206, r.values if r else [], 153, 46)
            p.text(
                x,
                274,
                f"{len(r.values)} channels" if r else p.tr("not_computed"),
                8,
                TEXT,
                mono=True,
            )
        p.text(
            48,
            312,
            f"{config.dim} → {config.hidden} → GELU → {config.dim}",
            8,
            MUTED,
            width=554,
        )
        return
    router = result_for(s, s.group.indices[0])
    experts = result_for(s, s.group.indices[1])
    data = router.details if router else {}
    weights = dict(zip(data.get("selected", []), data.get("weights", [])))
    count = min(4, config.experts)
    # Keep the small default layout stable. Larger banks prioritize selected
    # experts by weight, then use any remaining slots for idle experts.
    visible = (
        list(range(count))
        if config.experts <= count
        else (
            sorted(weights, key=lambda expert: -weights[expert])
            + [expert for expert in range(config.experts) if expert not in weights]
        )[:count]
    )
    step = 556 / count
    p.text(48, 159, p.tr("router", k=config.top_k), 15, GOLD)
    if config.experts > count:
        omitted = sum(
            weight for expert, weight in weights.items() if expert not in visible
        )
        label = (
            p.tr("expert_crop", shown=count, total=config.experts, rest=omitted)
            if router
            else p.tr("expert_crop_wait", shown=count, total=config.experts)
        )
        p.fit_text(48, 181, label, 552, 10, MUTED)
    for slot, idx in enumerate(visible):
        x = 49 + slot * step
        selected = idx in weights
        color = GOLD if selected else "#738a94"
        p.poly(
            [
                (x + 5, 195),
                (x + step - 37, 195),
                (x + step - 23, 209),
                (x + step - 23, 255),
                (x + 5, 255),
            ],
            "#203b4c",
        )
        p.line([(x + 5, 195), (x + step - 37, 195), (x + step - 23, 209)], color, 2)
        p.text(x + 13, 211, f"EXPERT {idx + 1}", 11, color, mono=True)
        p.text(
            x + 13,
            232,
            f"{config.dim}→{config.hidden}→{config.dim}",
            10,
            color,
            mono=True,
        )
        if experts and selected:
            bars(p, x + 7, 264, experts.details["experts"][idx], step - 30, 25)
        p.fit_text(
            x + 10,
            302,
            p.tr("weight", weight=weights[idx]) if selected else p.tr("sleeping"),
            step - 20,
            10,
            color,
        )


def normalization_world(p, s):
    index = s.group.indices[0]
    source = result_for(s, index - 1) if index else None
    result = result_for(s, index)
    p.text(65, 168, "RMSNorm / scale", 15, GOLD)
    bars(p, 67, 222, source.values if source else [], 211, 45)
    p.text(67, 283, p.tr("before_norm"), 10, TEXT)
    p.text(326, 231, "→", 24, GOLD)
    bars(p, 386, 222, result.values if result else [], 211, 45)
    p.text(386, 283, p.tr("after_norm"), 10, TEXT)
    p.text(65, 315, p.tr("norm_formula"), 12, MUTED)


def residual_world(p, s):
    result = result_for(s, s.group.indices[0])
    p.text(49, 160, "Residual / skip + branch", 15, GOLD)
    character(p, 103, 233, ghost=True)
    p.text(74, 254, p.tr("skip"), 9, "#9cbcc6")
    character(p, 290, 233)
    p.text(264, 254, p.tr("branch"), 9, TEXT)
    p.text(190, 201, "+", 24, GOLD)
    p.text(383, 201, "=", 24, GOLD)
    character(p, 490, 233)
    p.text(470, 254, p.tr("sum"), 9, GOLD)
    if result:
        for x, key in ((64, "skip"), (253, "branch")):
            bars(p, x, 281, result.details[key], 129, 29)
        bars(p, 444, 281, result.values, 129, 29)


DRAWERS = {
    "embedding": embedding_world,
    "attention": attention_world,
    "ffn": ffn_world,
    "norm_attn": normalization_world,
    "norm_ffn": normalization_world,
    "norm_final": normalization_world,
    "residual_attn": residual_world,
    "residual_ffn": residual_world,
    "unembedding": output_world,
    "output": output_world,
}


def draw_module(p, s):
    DRAWERS[s.group.house](p, s)
