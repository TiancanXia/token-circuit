"""Shared head selection controls for Attention module and child diagrams."""

from .painting import GOLD, TEXT

BLUE = "#87b8cc"


def head_tags(head):
    return ("attention-choice", "head-choice", f"probe:select_head:{head}")


def head_selector(p, s, head):
    config = s.forward.config
    p.fit_text(
        600,
        134,
        "N · " + p.tr("probe.head", head=head + 1, kv=config.kv_head_for(head) + 1),
        270,
        11,
        BLUE,
        anchor="ne",
        tags="head-indicator",
    )
    start = head // 4 * 4
    choices = [(-1, "‹", "head", -1)]
    choices += [
        (i - start, f"Q{i + 1}", "select_head", i)
        for i in range(start, min(start + 4, config.heads))
    ]
    choices += [(4, "›", "head", 1)]
    for slot, label, action, value in choices:
        x = 378 + slot * 44
        tags = ("attention-choice", "head-choice", f"probe:{action}:{value}")
        active = action == "select_head" and value == head
        p.rect(x, 154, 39, 25, "#46564d" if active else "#253e51", tags=tags)
        p.fit_text(
            x + 19.5,
            158,
            label,
            35,
            13,
            GOLD if active else TEXT,
            anchor="n",
            tags=tags,
        )
