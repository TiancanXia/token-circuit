"""Ordered toy decoder: full-prefix prefill or one-token cached decoding."""

from dataclasses import dataclass
import copy
import hashlib
import math
import random
from .architecture import ModelConfig, Stage, build_stages
from .mathops import rmsnorm, linear, rope, attention_trace, gelu, softmax, top_k


def matrix(rows, cols, name):
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    limit = 1 / math.sqrt(cols)
    return [[rng.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


@dataclass(frozen=True)
class Result:
    stage: Stage
    values: list
    label: str
    details: dict


class ForwardPass:
    def __init__(self, config=None, seed=17, cache=None):
        self.config = config or ModelConfig()
        self.cache = cache
        self.cache_requested = cache is not None
        self.cache_reused = (
            len(cache.context) if cache and cache.reusable(self.config) else 0
        )
        self.cached_layers = copy.deepcopy(cache.layers) if self.cache_reused else {}
        self.pending_layers = {}
        self.positions = (
            [len(self.config.context) - 1]
            if self.cache_reused
            else list(range(len(self.config.context)))
        )
        self.stages = build_stages(self.config)
        self.next_index = 0
        self.results = []
        self.x = []
        self.skip = []
        self.q = self.k = self.v = []
        self.routes = []
        self.expert_outputs = []
        self.probabilities = []
        self.sampled = None
        self.rng = random.Random(seed)
        self.embedding_table = matrix(
            len(self.config.vocab), self.config.dim, "embedding"
        )
        self.handlers = {
            "embedding": self.embedding,
            "norm_attn": self.norm,
            "norm_ffn": self.norm,
            "norm_final": self.norm,
            "qkv": self.qkv,
            "rope": self.rotate,
            "attention": self.attend,
            "project": self.project,
            "residual_attn": self.residual,
            "residual_ffn": self.residual,
            "up": self.up,
            "activation": self.activation,
            "down": self.down,
            "router": self.router,
            "experts": self.experts,
            "combine": self.combine,
            "unembedding": self.unembedding,
            "softmax": self.normalize_logits,
            "sample": self.sample,
        }

    @property
    def current_vector(self):
        return self.results[-1].values[:] if self.results else []

    def execute(self, index):
        if index != self.next_index or not 0 <= index < len(self.stages):
            raise ValueError("Each stage must execute once, in order")
        stage = self.stages[index]
        values, label, details = self.handlers[stage.kind](stage)
        result = Result(stage, values[:], label, copy.deepcopy(details))
        self.results.append(result)
        self.next_index += 1
        if stage.kind == "sample" and self.cache is not None:
            self.cache.publish(self.config, self.pending_layers)
        return result

    def preview(self, index):
        """A detached current-stage preview; neither execution nor cache is mutated."""
        if 0 <= index < self.next_index:
            return self.results[index]
        if index != self.next_index:
            raise ValueError("Only reached stages can be inspected")
        detached = copy.deepcopy(self)
        detached.cache = None
        return detached.execute(index)

    def project_rows(self, rows, out_dim, name):
        weights = matrix(out_dim, len(rows[0]), name)
        return [linear(row, weights) for row in rows]

    def attention_metadata(self):
        return {
            "positions": self.positions,
            "reused": self.cache_reused,
            "phase": "decode"
            if self.cache_reused
            else "prefill"
            if self.cache_requested
            else "full",
        }

    def embedding(self, stage):
        self.x = [
            self.embedding_table[self.config.context[pos]][:] for pos in self.positions
        ]
        return (
            self.x[-1],
            "token embedding",
            {"context": self.config.context, "matrix": self.x},
        )

    def norm(self, stage):
        before = self.x[-1][:]
        if stage.kind != "norm_final":
            self.skip = copy.deepcopy(self.x)
        self.x = [rmsnorm(row) for row in self.x]
        return (
            self.x[-1],
            "normalized stream",
            {
                "before": before,
                "rms": math.sqrt(sum(v * v for v in before) / len(before)),
            },
        )

    def qkv(self, stage):
        base = f"b{stage.block}"
        source = copy.deepcopy(self.x)
        weights = {
            name: matrix(
                self.config.dim if name == "q" else self.config.kv_dim,
                self.config.dim,
                base + name,
            )
            for name in ("q", "k", "v")
        }
        self.q, self.k, self.v = [
            [linear(row, weights[name]) for row in source] for name in ("q", "k", "v")
        ]
        return (
            self.q[-1],
            "Q (current token)",
            {
                "q": self.q[-1],
                "k": self.k[-1],
                "v": self.v[-1],
                "q_matrix": self.q,
                "k_matrix": self.k,
                "v_matrix": self.v,
                "input": source[-1],
                "weights": weights,
                **self.attention_metadata(),
            },
        )

    def rotate(self, stage):
        width = self.config.head_dim
        before_q, before_k = copy.deepcopy(self.q), copy.deepcopy(self.k)

        def rotate_rows(rows):
            return [
                [
                    value
                    for start in range(0, len(row), width)
                    for value in rope(row[start : start + width], pos)
                ]
                for pos, row in zip(self.positions, rows)
            ]

        self.q, self.k = rotate_rows(self.q), rotate_rows(self.k)
        old = self.cached_layers.get(stage.block, {"k": [], "v": []})
        self.k, self.v = old["k"] + self.k, old["v"] + self.v
        self.pending_layers[stage.block] = {
            "k": copy.deepcopy(self.k),
            "v": copy.deepcopy(self.v),
        }
        return (
            self.q[-1],
            "rotated Q",
            {
                "q": self.q[-1],
                "k": self.k[-1],
                "v": self.v[-1],
                "position": self.positions[-1],
                "q_matrix": self.q,
                "k_matrix": self.k,
                "v_matrix": self.v,
                "before_q": before_q[-1],
                "before_k": before_k[-1],
                **self.attention_metadata(),
            },
        )

    def attend(self, stage):
        self.x, traces = attention_trace(
            self.q,
            self.k,
            self.v,
            self.config.heads,
            self.config.num_kv_heads,
            self.positions,
        )
        weights = [trace["weights"] for trace in traces]
        return (
            self.x[-1],
            "concatenated heads",
            {
                "attention": [head[-1] for head in weights],
                "all_attention": weights,
                "traces": traces,
                "q_matrix": self.q,
                "k_matrix": self.k,
                "v_matrix": self.v,
                **self.attention_metadata(),
            },
        )

    def project(self, stage):
        before = self.x[-1][:]
        weights = matrix(self.config.dim, self.config.dim, f"b{stage.block}out")
        self.x = [linear(row, weights) for row in self.x]
        return (
            self.x[-1],
            "attention branch",
            {"input": before, "weights": weights, **self.attention_metadata()},
        )

    def residual(self, stage):
        before, branch = self.skip[-1][:], self.x[-1][:]
        self.x = [
            [a + b for a, b in zip(original, delta)]
            for original, delta in zip(self.skip, self.x)
        ]
        return self.x[-1], "residual stream", {"skip": before, "branch": branch}

    def up(self, stage):
        self.x = self.project_rows(self.x, self.config.hidden, f"b{stage.block}up")
        return (
            self.x[-1],
            "expanded features",
            {"shape": (self.config.dim, self.config.hidden)},
        )

    def activation(self, stage):
        self.x = [[gelu(v) for v in row] for row in self.x]
        return self.x[-1], "GELU features", {}

    def down(self, stage):
        self.x = self.project_rows(self.x, self.config.dim, f"b{stage.block}down")
        return (
            self.x[-1],
            "FFN branch",
            {"shape": (self.config.hidden, self.config.dim)},
        )

    def router(self, stage):
        scores = self.project_rows(self.x, self.config.experts, f"b{stage.block}router")
        self.routes = [top_k(row, self.config.top_k) for row in scores]
        selected, weights = self.routes[-1]
        return (
            self.x[-1],
            "router input",
            {"scores": scores[-1], "selected": selected, "weights": weights},
        )

    def experts(self, stage):
        self.expert_outputs = []
        for row, (selected, _) in zip(self.x, self.routes):
            outputs = {}
            for idx in selected:
                name = f"b{stage.block}expert{idx}"
                hidden = linear(
                    row, matrix(self.config.hidden, self.config.dim, name + "up")
                )
                outputs[idx] = linear(
                    [gelu(v) for v in hidden],
                    matrix(self.config.dim, self.config.hidden, name + "down"),
                )
            self.expert_outputs.append(outputs)
        selected, weights = self.routes[-1]
        return (
            self.x[-1],
            "expert inputs",
            {
                "experts": self.expert_outputs[-1],
                "selected": selected,
                "weights": weights,
                "shape": (self.config.dim, self.config.hidden, self.config.dim),
            },
        )

    def combine(self, stage):
        self.x = [
            [
                sum(weight * outputs[idx][d] for idx, weight in zip(selected, weights))
                for d in range(self.config.dim)
            ]
            for outputs, (selected, weights) in zip(self.expert_outputs, self.routes)
        ]
        selected, weights = self.routes[-1]
        return (
            self.x[-1],
            "weighted expert output",
            {
                "experts": self.expert_outputs[-1],
                "selected": selected,
                "weights": weights,
            },
        )

    def unembedding(self, stage):
        logits = linear(
            self.x[-1], matrix(len(self.config.vocab), self.config.dim, "unembedding")
        )
        self.logits = logits
        return logits, "next-token logits", {"logits": logits}

    def normalize_logits(self, stage):
        self.probabilities = softmax(self.logits)
        return (
            self.probabilities,
            "next-token probabilities",
            {"probabilities": self.probabilities},
        )

    def sample(self, stage):
        draw = self.rng.random()
        cumulative, selected = 0.0, len(self.probabilities) - 1
        for idx, p in enumerate(self.probabilities):
            cumulative += p
            if draw < cumulative:
                selected = idx
                break
        self.sampled = self.config.vocab[selected]
        return (
            self.probabilities,
            "sample distribution",
            {
                "probabilities": self.probabilities,
                "sampled": self.sampled,
                "draw": draw,
            },
        )
