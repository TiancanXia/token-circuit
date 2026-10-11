"""Bilingual attention handbook, progressing from intuition to the equations."""

FORMULA_TITLES = (
    ("Attention scores · 注意力分数", "Attention scores"),
    ("Causal mask / Softmax", "Causal mask / Softmax"),
    ("Weighted sum / Output projection", "Weighted sum / Output projection"),
)
FORMULA_SECTION = ("公式与符号", "Formulas and symbols")

SECTIONS = (
    (
        ("Attention 做什么", "What Attention does"),
        (
            "让当前 token 汇集前文的信息。\n\nQ（Query）：用于查询。\nK（Key）：用于与 Q 匹配。\nV（Value）：携带要汇集的信息。\n\nQ 与各个 K 的匹配分数，经 Softmax 变成注意力权重，再用这些权重对 V 加权求和。",
            "Let the current token gather information from earlier positions.\n\nQ (Query): the query.\nK (Key): what Q matches against.\nV (Value): the information to combine.\n\nSoftmax turns Q–K matching scores into attention weights. These weights determine the weighted sum of V.",
        ),
    ),
    (
        ("一个小例子", "A small example"),
        (
            "假设两个位置的权重是 25% 和 75%，V 分别是 2 和 6。\n\n加权求和：0.25 × 2 + 0.75 × 6 = 5。\n\n实际的 V 是向量，每个通道都这样计算。权重大不代表贡献一定大，还要看 V 的数值。",
            "Suppose two positions have weights of 25% and 75%, and V values of 2 and 6.\n\nWeighted sum: 0.25 × 2 + 0.75 × 6 = 5.\n\nReal V values are vectors; each channel follows this calculation. A larger weight need not mean a larger contribution: V matters too.",
        ),
    ),
    (
        ("怎么看图", "Reading the diagram"),
        (
            "Query 是正在查看的查询。卡片在 Scores 处显示匹配分数，在 Softmax 处显示注意力权重；百分比不是下一个 token 的概率。\n\n点击卡片高亮对应连线。Query 箭头只在有多行可查看时出现，Q→KV 按钮或 N 切换 head。长前缀可以翻页；权重始终按全部位置计算。\n\n向量用色条表示：金色为正，蓝色为负，每组色条独立缩放。Concat 把各个 head 的输出接在一起，W_o 再混合这些特征。",
            "Query is the query being inspected. Cards show matching scores at Scores and attention weights at Softmax. The percentages are not next-token probabilities.\n\nClick a card to highlight its connection. Query arrows appear only when multiple rows are available. Q→KV or N changes the head. Long prefixes have pages; weights always use all positions.\n\nVector bars use gold for positive values and blue for negative values, with each group scaled separately. Concat joins the head outputs; W_o then mixes their features.",
        ),
    ),
    (
        FORMULA_SECTION,
        (
            "t 是 query 位置，j 是 key 位置，h 是查询头，g(h) 是它对应的 KV head，d 是一个 head 的维度。\n\n先计算 Q、K 的点积并除以 √d；需要时屏蔽未来位置。Softmax 沿 key 位置计算，权重合计为 1。o 是一个 head 的结果，y 是输出投影后的结果。",
            "t is the query position, j the key position, h the query head, g(h) its KV head, and d the width of one head.\n\nCompute the Q–K dot product and divide by √d; mask future positions when needed. Softmax runs across key positions, with weights summing to 1. o is one head's result; y is the projected output.",
        ),
    ),
    (
        ("MHA 与 GQA", "MHA and GQA"),
        (
            "MHA：每个 Q head 有自己的 K/V head。\nGQA：一组 Q head 共用一个 K/V head。\n\n共享的是 K/V；每个 Q head 仍独立计算分数、权重和输出。当前有 {heads} 个 Q head、{kv} 个 KV head。\n\n所有 Q head 的结果拼接一次，W_o 将它投影回 {dim} 维，再进入残差连接。注意力权重不能直接当作语义解释或因果重要性。",
            "MHA: each Q head has its own K/V head.\nGQA: a group of Q heads shares one K/V head.\n\nOnly K/V is shared. Each Q head still computes its own scores, weights and output. This model has {heads} Q heads and {kv} KV heads.\n\nConcat joins all Q head outputs once. W_o projects them back to {dim} channels before the residual connection. Attention weights are not a direct semantic explanation or measure of causal importance.",
        ),
    ),
    (
        ("KV cache 与 Causal mask", "KV cache and Causal mask"),
        (
            "KV cache 保存每层已算过的 K/V，后续 token 可以复用它们。保存的是 RoPE 后的 K 和未旋转的 V，不缓存 Q。\n\nCausal mask 保证查询看不到未来位置。单 token 解码时，已有 key 都在当前查询之前或同一位置，因此可以跳过显式屏蔽。\n\n预填充的较早查询、一次追加多个 token 时仍要屏蔽未来 key。打开缓存不会取消因果约束。",
            "KV cache stores each layer's computed K/V for later tokens to reuse. It stores K after RoPE and unrotated V, not Q.\n\nCausal mask prevents a query from seeing future positions. In single-token decode, all existing keys are at or before the query, so explicit masking can be skipped.\n\nEarlier prefill queries and multi-token chunks still need future keys masked. Enabling cache does not remove causality.",
        ),
    ),
)


def sections(config, language):
    index = language == "en"
    return tuple(
        (
            heading[index],
            body[index].format(
                heads=config.heads,
                kv=config.num_kv_heads,
                head_dim=config.head_dim,
                dim=config.dim,
            ),
        )
        for heading, body in SECTIONS
    )
