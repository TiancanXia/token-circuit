"""Vocabulary projections and full-softmax probabilities with a Top-N display."""

from .painting import GOLD, TEXT, MUTED


def output_summary(values, limit=7):
    """Keep original values; display cropping must never renormalize probabilities."""
    chosen = tuple(
        sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:limit]
    )
    remainder = sum(value for i, value in enumerate(values) if i not in chosen)
    return chosen, remainder


def output_world(p, s):
    keys = {s.forward.stages[i].key for i in s.group.indices}
    result = next(
        (
            r
            for r in reversed(s.forward.results)
            if r.stage.key in keys
            and ("probabilities" in r.details or "logits" in r.details)
        ),
        None,
    )
    probs = s.group.house == "output"
    total = len(s.forward.config.vocab)
    p.text(
        49,
        164,
        p.tr("output_probs_title" if probs else "output_logits_title", n=total),
        15,
        GOLD,
    )
    if not result:
        p.text(49, 211, p.tr("probability_hint" if probs else "vocab_hint"), 10, MUTED)
        return
    values = result.details["probabilities" if probs else "logits"]
    chosen, remainder = output_summary(values)
    for col, idx in enumerate(chosen):
        x = 71 + col * 78
        p.rect(x - 14, 209, 28, 43, "#29424b")
        p.rect(x - 10, 215, 20, 31, "#839581")
        p.fit_text(x, 183, s.forward.config.vocab[idx], 72, 15, TEXT, anchor="n")
        p.text(
            x,
            272,
            f"{values[idx]:.1%}" if probs else f"{values[idx]:+.2f}",
            11,
            GOLD,
            anchor="n",
            mono=True,
        )
    summary = (
        p.tr("top_probabilities", n=len(chosen), total=total, rest=remainder)
        if probs
        else p.tr("top_logits", n=len(chosen), total=total)
    )
    p.text(49, 304, summary, 11, GOLD, width=558)
    note = (
        p.tr("journey_probs" if s.guided else "full_sample")
        if probs
        else p.tr("logits_note")
    )
    p.text(49, 322, note, 11, MUTED, width=558)
