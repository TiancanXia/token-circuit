# 模型计算

[English](model-audit.md) | [简体中文](model-audit.zh-CN.md) · [首页](../README.zh-CN.md)

两层 pre-norm causal decoder，参数固定且未训练。第一层使用 Dense FFN，第二层使用 MoE。

## 前向过程

```text
Embedding
  每个 block:
    skip = x
    RMSNorm → QKV → RoPE(Q, K) → head attention → Concat → W_o
    x = skip + attention output
    skip = x
    RMSNorm → Dense FFN 或 MoE
    x = skip + FFN output
Final RMSNorm → 最后位置 → Unembedding → Softmax → Sample
```

## 默认数值

| 配置 | 数值 |
| --- | --- |
| 隐藏维度 | 8 |
| Q heads / head 维度 | 4 / 2 |
| MHA / GQA 的 KV heads | 4 / 2 |
| FFN 中间维度 | 12 |
| MoE | 每个 token 从 3 个专家选 2 个 |
| 词表 | 中文 16，英文 12 |
| 上下文上限 | 64 token |

通用 `ModelConfig` 默认 2 个 Q head，游戏配置使用 4 个。

## 算子

- **RMSNorm：** x / √(mean(x²)+10⁻⁶)，γ=1，不减均值；残差输入在归一化前保存。
- **QKV：** 对归一化输入做独立线性投影。矩阵按输出×输入存储，行向量公式为 XWᵀ。
- **RoPE：** 在每个 head 内旋转 Q/K 的成对分量，第 i 对角度为 pos / 10000^(2i/d_head)，V 不变。
- **Attention：** QKᵀ / √d_head → 因果可见性 → softmax → 加权 V。每个 Q head 独立输出，Concat 在 W_o 前只拼接一次。
- **GQA：** 4 个 Q head 共享 2 组 K/V，不对 query 输出求平均，拼接后仍为 8 维。
- **FFN：** 每个位置独立 Up(8→12) → GELU → Down(12→8)。
- **MoE：** 根据路由 logits 选 2 个专家，对所选 logits 做 softmax，再加权合并独立专家 FFN。
- **输出：** 最终归一化后，将最后位置投影到全词表 logits，使用 T=1 softmax 和带种子的抽样。

## 缓存与因果性

每层缓存旋转后的 K 与未旋转的 V。预填充计算完整前缀；兼容的单 token 解码只投影新位置，保留绝对位置。旧 K 不重复旋转。

最后有效 key 位置的单 query 没有未来 key，无需显式 mask。预填充的较早 query 和多 token 解码块仍需屏蔽未来位置，不能仅按缓存开关判断。

一轮完整计算统一提交所有层缓存。变更、非追加或截断上下文会重算；截断保留末尾 64 个 token，位置从零重算。图表裁剪不影响计算。

## 句子与概率

游戏句子与模型抽样结果独立。中文按字，英文按词与标点；单独的 `Session(guided=False)` 模式使用实际抽样。

Top 7 展示全词表概率最高的七项，并标注剩余概率；不重新归一化，也不把抽样限制在可见项。

参数未训练。不包含训练、bias、dropout、可学习 RMS γ、专家容量或负载均衡；输入 embedding 与输出矩阵未绑定。

## 验证与参考

测试覆盖因果输出、RoPE、残差、专家合并、GQA 与显式扩展 K/V 对照，以及缓存解码与完整前缀对照。命令见 [开发与贡献](../CONTRIBUTING.zh-CN.md)。

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RMSNorm](https://arxiv.org/abs/1910.07467)
- [RoFormer](https://arxiv.org/abs/2104.09864)
- [稀疏专家网络](https://arxiv.org/abs/1701.06538)
- [Llama 2](https://arxiv.org/abs/2307.09288)
- [GQA](https://arxiv.org/abs/2305.13245)
- [KV cache](https://huggingface.co/docs/transformers/main/en/cache_explanation)
