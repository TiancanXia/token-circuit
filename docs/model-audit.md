# Model computation

[English](model-audit.md) | [简体中文](model-audit.zh-CN.md) · [Documentation](README.md)

A two-block pre-norm causal decoder with fixed, untrained weights. Block 1 uses Dense FFN; block 2 uses MoE.

## Forward pass

```text
Embedding
  each block:
    skip = x
    RMSNorm → QKV → RoPE(Q, K) → head attention → Concat → W_o
    x = skip + attention output
    skip = x
    RMSNorm → Dense FFN or MoE
    x = skip + FFN output
Final RMSNorm → last position → Unembedding → Softmax → Sample
```

## Default values

| Setting | Value |
| --- | --- |
| Hidden width | 8 |
| Q heads / head width | 4 / 2 |
| MHA / GQA KV heads | 4 / 2 |
| FFN hidden width | 12 |
| MoE | 2 of 3 experts per token |
| Vocabulary | Chinese 16; English 12 |
| Context limit | 64 tokens |

The generic `ModelConfig` defaults to 2 Q heads; the game config uses 4.

## Operations

- **RMSNorm:** x / √(mean(x²)+10⁻⁶), with γ=1 and no mean subtraction. Residual input is saved before normalization.
- **QKV:** separate linear projections of normalized input. Matrices are stored output×input; row-vector notation is XWᵀ.
- **RoPE:** rotate Q/K pairs inside each head, using angle pos / 10000^(2i/d_head) for pair i. V is unchanged.
- **Attention:** QKᵀ / √d_head → causal visibility → softmax → weighted V. Each Q head produces its own output; Concat joins them once before W_o.
- **GQA:** 4 Q heads share 2 K/V groups. Query outputs are not averaged; the concatenated width remains 8.
- **FFN:** per-position Up(8→12) → GELU → Down(12→8).
- **MoE:** select 2 experts from router logits, softmax the selected logits, then merge independent expert FFNs by their weights.
- **Output:** final normalization, last-position projection to full-vocabulary logits, T=1 softmax and seeded sampling.

## Cache and causality

Each layer caches rotated K and unrotated V. Prefill processes the whole prefix; compatible single-token decoding projects only the new position, retaining its absolute position. Old K is not rotated again.

A single query at the last valid key has no future keys to mask. Earlier prefill queries and multi-token decoding chunks still mask future keys. Cache on/off alone does not determine masking.

The full pass commits all layers' caches together. Changed, non-append or truncated contexts are recomputed. Truncation keeps the last 64 tokens and restarts positions at zero. Diagram cropping does not affect computation.

## Sentence and probabilities

The playable sentence is independent of the model's sampled token. Chinese uses character tokens; English uses words and punctuation. The separate `Session(guided=False)` mode uses the actual sample.

Top 7 shows the highest full-vocabulary probabilities and labels omitted mass; it does not renormalize or limit sampling to the displayed items.

Weights are untrained. There is no training, bias, dropout, learned RMS γ, expert capacity/load-balancing or tied embedding/output matrix.

## Verification and references

Tests check causal outputs, RoPE, residuals, expert merging, GQA against expanded K/V and cached decoding against full-prefix computation. Commands are in [contributing](../CONTRIBUTING.md).

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RMSNorm](https://arxiv.org/abs/1910.07467)
- [RoFormer](https://arxiv.org/abs/2104.09864)
- [Sparse mixture of experts](https://arxiv.org/abs/1701.06538)
- [Llama 2](https://arxiv.org/abs/2307.09288)
- [GQA](https://arxiv.org/abs/2305.13245)
- [KV cache](https://huggingface.co/docs/transformers/main/en/cache_explanation)
