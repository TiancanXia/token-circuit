# 微型 Transformer 核对

[English](model-audit.md) | [简体中文](model-audit.zh-CN.md) · [文档目录](README.zh-CN.md)

结论：现有计算是成立的简化 **pre-norm causal decoder Transformer**。两层 block 分别用 Dense FFN 与 MoE；大地图的重复框仅压缩显示，每层都实际执行。它是未训练的教学模型，不是某个商业模型或 2017 年完整 encoder–decoder 的复刻。

## 一轮计算

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

| 核对项 | 实现与结果 |
|---|---|
| 维度 | 游戏默认 d_model=8，Q heads=4，d_head=2（偶数）；MHA KV heads=4、GQA KV heads=2；通用 ModelConfig 默认仍为 2 heads；FFN hidden=12 |
| RMSNorm | x / √(mean(x²)+10⁻⁶)，不减均值；γ 固定 1；残差保存发生在 norm 之前 |
| Q/K/V | 同一归一化输入的独立线性投影；预填充计算全前缀，缓存解码只投影新位置；GQA 的 K/V 宽度为 KV heads × d_head |
| RoPE | 每个 head 内成对旋转 Q/K；第 i 对角度为 pos / 10000^(2i/d_head)，i 从 0 开始；V 不变 |
| Attention | 每个 Q head 使用对应 KV head 计算 QKᵀ / √d_head；实际存在的未来位置设为 −∞；没有未来 key 的末尾 query 直接进入 softmax，再加权 V |
| 输出投影 | 先拼接 heads 再 W_o，输出维度与原残差流相同 |
| Dense FFN | 每个位置独立 Up(8→12) → GELU → Down(12→8)；扩展特征维度，不是图像上采样 |
| MoE | 每个 token 独立路由，3 专家选 2；对所选 logits 做 softmax；各专家独立 Up/GELU/Down 后加权合并 |
| 末端 | Final RMSNorm 后仅把最后一个位置映射到全词表 logits，T=1 softmax，然后进行带种子的随机采样 |
| 自回归 | 每次只输入已经亮起的前缀；一轮完毕才追加新 token；可逐层缓存历史 K/V，后续仅计算新 token；完整前缀对照概率一致 |

注意力算子在逐头汇聚之后拼接全部 Q head；界面将唯一一次 Concat 放在 Head attention 的最后一个内部检查点。后续 Output · W_o 的 Input 只查看这份已拼接输入。矩阵在代码中存为“输出维度 × 输入维度”，行向量投影写作 XWᵀ。

MHA 与 GQA 是整层 Attention 的 head 对应方式：MHA 的 4 个 Q head 分别读取 4 组 K/V；GQA 的 4 个 Q head 分为两组、读取 2 组 K/V。每个 Q head 仍独立产生 2 维输出，最终始终拼接成 8 维，不在组内求平均或减少 Q 输出数。分组在 QKV 的 Heads 处展示，真正的逐头注意力位于 RoPE/读取历史 K/V 之后、W_o 之前。

缓存开启并不自动取消因果约束。当前 demo 的单 token decode 使用无 padding、无预分配空槽的动态有效 K/V，query 位于最后一个 key 的位置，因而无需构造显式因果 mask。预填充有多个 query，较早 query 仍不能读取后续 key；一次追加多个新 token 的分块 decode 也必须屏蔽块内未来位置。算子的 mask_applied 按实际 query 位置标记，视图直接读取这些诊断，不按缓存开关猜测。

## 有意保留的简化

固定、未训练且可复现的权重；不含 bias、训练、dropout、可学习的 RMS γ、专家负载均衡或容量限制。输入 embedding 与输出矩阵未绑定。GELU 是合法的 FFN 选择，但例如 Llama 使用 SwiGLU，所以不能把这里称为 Llama 的精确架构。

当前句子采用预设 token，计算出的真实概率单独展示；预设下一 token 不是模型采样结果。底层自由采样模式仍使用实际抽样。中文按字、英文按词/标点切分，均为教学用规则。长上下文截断到末 64 token 后位置从 0 重算；图表另有裁剪，完整计算不受裁剪影响。

## 可验证性

数值测试覆盖 RMSNorm 不去均值、RoPE 范数和相对位置性质、因果 attention、原始残差、专家权重/合并、QKV 诊断快照。新增整网因果测试：只改变后两个输入 token，经过 Dense 与 MoE 两层及最终 norm 后，前两个位置的输出保持完全相同，后两个位置发生变化。

游戏测试验证重复 block 真正执行、手动与快速 forward 一致、一次转场只追加一次 token，以及句末停止。中英文共用同一数值实现。

Attention 测试覆盖末尾单 query、多 query 分块因果约束、MHA/GQA head 对应、唯一 Concat、预填充三角遮罩，以及等待节点的数据隔离。运行 `python -m unittest discover -s tests -v` 获取当前验证结果，运行 `python check_ui.py` 检查两种语言的界面交互。

缓存/GQA 测试将分组 K/V 与显式复制成 MHA 的参考计算对照，并将逐 token 缓存解码与完整前缀计算对照。缓存必须按层区分，旧 K 不再次施加 RoPE，位置沿用完整前缀下标；截断后全量重建。子地图预览不会提交模型状态或缓存，完整 forward 后才统一发布 K/V。

## 原始参考

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：多头注意力、缩放点积、前馈和残差。
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)：RMSNorm 的定义。
- [RoFormer](https://arxiv.org/abs/2104.09864)：旋转位置编码。
- [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538)：稀疏专家路由。
- [Llama 2](https://arxiv.org/abs/2307.09288)，§2.2：采用 pre-norm RMSNorm、RoPE 与 SwiGLU，供架构差异对照。
- [GQA](https://arxiv.org/abs/2305.13245)：Q head 分组共享较少的 K/V head。
- [Transformers cache explanation](https://huggingface.co/docs/transformers/main/en/cache_explanation)：自回归推理中历史 K/V 的复用。
