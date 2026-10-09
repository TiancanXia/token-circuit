"""Committed, per-layer inference cache. Partial forwards never change this bank."""

import copy
from dataclasses import dataclass, field


def signature(config):
    return (
        config.dim,
        config.heads,
        config.num_kv_heads,
        config.hidden,
        config.experts,
        config.top_k,
        config.blocks,
        config.vocab,
    )


@dataclass
class KVCache:
    context: tuple = ()
    model_signature: tuple = ()
    layers: dict = field(default_factory=dict)

    def reusable(self, config):
        return (
            bool(self.context)
            and self.context == config.context[:-1]
            and self.model_signature == signature(config)
            and set(self.layers) == set(range(len(config.blocks)))
        )

    def publish(self, config, layers):
        if set(layers) != set(range(len(config.blocks))):
            raise ValueError("Only a complete forward can publish its cache")
        self.layers = copy.deepcopy(layers)
        self.context = config.context
        self.model_signature = signature(config)
