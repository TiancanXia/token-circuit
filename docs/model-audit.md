# Miniature Transformer model notes

[English](model-audit.md) | [简体中文](model-audit.zh-CN.md) · [Documentation](README.md)

The computation implements a simplified **pre-norm causal decoder Transformer**. Its two blocks use Dense FFN and MoE respectively. The repeated-block frame only compresses the map: every layer actually executes. This is an untrained educational model, not a commercial model or a replica of the full 2017 encoder–decoder.

## One forward pass

```text
Embedding
  for each block:
    skip = x
    RMSNorm(x) → Q, K, V → per-head RoPE(Q, K)
    per-Q-head scores → visibility → Softmax → weighted V → Concat (once) → W_o
    x = skip + attention branch
    skip = x
    RMSNorm(x) → Dense FFN OR tokenwise MoE
    x = skip + FFN branch
Final RMSNorm → last token → Unembedding → Softmax → Sample
```

| Component | Implementation |
| --- | --- |
| Dimensions | Game defaults: d_model=8, 4 Q heads, even d_head=2; MHA has 4 KV heads, GQA has 2; the generic ModelConfig still defaults to 2 heads. FFN hidden width is 12. |
| RMSNorm | x / √(mean(x²)+10⁻⁶), without mean subtraction; γ is fixed at 1. Residual streams are saved before normalization. |
| Q/K/V | Independent linear projections of the same normalized input. Prefill projects the full prefix; cached decoding projects only new positions. GQA K/V width equals KV heads × d_head. |
| RoPE | Rotate Q/K pairs inside each head; pair i uses angle pos / 10000^(2i/d_head), starting at i=0. V is unchanged. |
| Attention | Each Q head reads its assigned KV head, using QKᵀ / √d_head. Actual future positions receive −∞. A final query with no future keys proceeds directly to softmax and a weighted V sum. |
| Output projection | Concatenate heads once, then apply W_o; output width matches the residual stream. |
| Dense FFN | Per-position Up(8→12) → GELU → Down(12→8). This expands feature width, not image resolution. |
| MoE | Route each token independently to 2 of 3 experts. Softmax the selected router logits; each expert has its own Up/GELU/Down path, followed by a weighted merge. |
| Output | Final RMSNorm, then project the last position to full-vocabulary logits. T=1 softmax produces probabilities, followed by seeded random sampling. |
| Autoregression | Input contains only the revealed prefix. Append a token only after a full pass. Per-layer K/V can be reused, computing only the new token; probabilities agree with full-prefix computation. |

The numerical operator concatenates all Q-head outputs after head attention. Its unique Concat checkpoint is inside Head attention; the later Output · W_o input checkpoint inspects the already-packed vector. Matrices are stored as output-width × input-width, so row-vector projections use XWᵀ.

MHA/GQA configure the entire Attention layer. In MHA, four Q heads read four separate K/V groups. In GQA, the same four Q heads form two groups sharing two K/V groups. Every Q head still produces an independent two-dimensional output; Concat remains eight-dimensional. Query outputs are not averaged within a group. QKV displays the mapping; head attention follows RoPE/history preparation and precedes W_o.

Enabling cache does not remove causality. Single-token decoding uses dynamically sized valid K/V, without padding or preallocated empty slots. The query is at the last key position, so no explicit causal mask is needed. Earlier prefill queries still mask later keys; chunked multi-token decoding must also mask future positions within the chunk. Views read per-query `mask_applied` diagnostics rather than guessing from the cache switch.

## Deliberate simplifications

Weights are fixed, untrained and reproducible. There is no bias, training, dropout, learned RMS γ, expert load balancing or capacity limit. Embedding and output weights are not tied. GELU is a valid FFN choice, but architectures such as Llama use SwiGLU; this is not an exact Llama implementation.

The journey uses prescribed tokens and shows independently computed probabilities. Its next token is not the model's sampled token. The separate free-sampling mode uses the actual draw. Chinese uses characters; English uses words and punctuation, both demonstration rules rather than production tokenization. Contexts longer than 64 tokens retain the last 64 and recompute positions from zero. Diagram cropping does not change the full computation.

## Verification

Numerical tests cover RMSNorm without mean subtraction, RoPE norm/relative-position properties, causal attention, original residual streams, expert weights/merging and QKV snapshots. Whole-network causality is checked by changing only the last two inputs: earlier positions remain identical after Dense/MoE blocks and final norm, while later positions change.

Game tests check actual execution of repeated blocks, manual/fast-pass equivalence, one append per transfer and stopping at the sentence end. Both languages share one numerical implementation. Attention tests cover final single queries, chunk causality, MHA/GQA mapping, unique Concat, prefill masks and isolation of pending-stage displays.

Cache/GQA tests compare shared K/V against explicitly expanded MHA and incremental decoding against full-prefix computation. Caches are layer-specific; old K is not rotated twice, absolute prefix positions are retained, and truncation rebuilds the full computation. Child previews do not publish state/cache; completed passes publish all layers together.

```sh
python -m unittest discover -s tests -v
python check_ui.py
```

## Primary references

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): multi-head/scaled dot-product attention, FFNs and residuals.
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467): RMSNorm.
- [RoFormer](https://arxiv.org/abs/2104.09864): rotary position encoding.
- [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538): sparse expert routing.
- [Llama 2](https://arxiv.org/abs/2307.09288), §2.2: pre-norm RMSNorm, RoPE and SwiGLU, for comparison.
- [GQA](https://arxiv.org/abs/2305.13245): grouped queries sharing fewer K/V heads.
- [Transformers cache explanation](https://huggingface.co/docs/transformers/main/en/cache_explanation): reusing historical K/V during autoregressive inference.
