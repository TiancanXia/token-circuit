"""Small pure-Python numerical operators; no window or game state."""

import math


def softmax(values):
    maximum = max(values)
    weights = [math.exp(x - maximum) for x in values]
    total = sum(weights)
    return [x / total for x in weights]


def rmsnorm(values, epsilon=1e-6):
    scale = math.sqrt(sum(x * x for x in values) / len(values) + epsilon)
    return [x / scale for x in values]


def linear(values, matrix):
    return [sum(x * w for x, w in zip(values, row)) for row in matrix]


def rope(values, position, base=10000.0):
    if len(values) % 2:
        raise ValueError("RoPE requires even head dimensions")
    result = []
    for i in range(0, len(values), 2):
        angle = position / base ** (i / len(values))
        c, s = math.cos(angle), math.sin(angle)
        result.extend(
            (values[i] * c - values[i + 1] * s, values[i] * s + values[i + 1] * c)
        )
    return result


def attention_trace(q, k, v, heads, kv_heads=None, query_positions=None):
    """One implementation for full-prefix MHA/GQA and incremental decoding."""
    kv_heads = heads if kv_heads is None else kv_heads
    if not q or not k or len(k) != len(v) or heads <= 0 or kv_heads <= 0:
        raise ValueError("Invalid attention shape")
    dim = len(q[0])
    if dim % heads or heads % kv_heads:
        raise ValueError("Invalid head grouping")
    width = dim // heads
    if any(len(row) != dim for row in q) or any(
        len(row) != width * kv_heads for row in k + v
    ):
        raise ValueError("Inconsistent Q/K/V widths")
    positions = (
        list(range(len(q))) if query_positions is None else list(query_positions)
    )
    if len(positions) != len(q) or any(not 0 <= pos < len(k) for pos in positions):
        raise ValueError("Query positions must address the key sequence")
    output = [[] for _ in q]
    traces = []
    for head in range(heads):
        kv_head = head // (heads // kv_heads)
        start, ks = head * width, kv_head * width
        rows, scores_all, masked_all, outputs, mask_applied = [], [], [], [], []
        for i, pos in enumerate(positions):
            scores = [
                sum(
                    a * b
                    for a, b in zip(q[i][start : start + width], row[ks : ks + width])
                )
                / math.sqrt(width)
                for row in k
            ]
            needs_mask = pos < len(k) - 1
            masked = (
                [value if j <= pos else -math.inf for j, value in enumerate(scores)]
                if needs_mask
                else scores[:]
            )
            probs = softmax(masked)
            values = [
                sum(probs[j] * v[j][d] for j in range(len(v)))
                for d in range(ks, ks + width)
            ]
            output[i].extend(values)
            rows.append(probs)
            scores_all.append(scores)
            masked_all.append(masked)
            outputs.append(values)
            mask_applied.append(needs_mask)
        traces.append(
            {
                "kv_head": kv_head,
                "scores": scores_all,
                "masked": masked_all,
                "weights": rows,
                "outputs": outputs,
                "mask_applied": mask_applied,
            }
        )
    return output, traces


def attention(q, k, v, heads, kv_heads=None, query_positions=None):
    output, traces = attention_trace(q, k, v, heads, kv_heads, query_positions)
    return output, [trace["weights"] for trace in traces]


def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


def top_k(logits, count):
    if not 1 <= count <= len(logits):
        raise ValueError("Invalid expert count")
    chosen = sorted(range(len(logits)), key=lambda i: (-logits[i], i))[:count]
    return chosen, softmax([logits[i] for i in chosen])
