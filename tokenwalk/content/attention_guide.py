"""Layered bilingual attention explanation; only reads configuration."""

FORMULA_TITLES = (
    ("匹配分数", "Match score"),
    ("可见位置的概率", "Causal probabilities"),
    ("汇聚内容与输出投影", "Gather and project"),
)

SECTIONS = (
    (
        ("直觉", "Intuition"),
        (
            "Q 是当前查询，K 提供匹配线索，V 携带要汇聚的内容。匹配分数经过 softmax 变成权重，再对 V 加权求和。较大的权重不一定带来较大的输出：V 的大小和正负也会影响贡献。",
            "Q is the query, K provides matching features and V carries content. Softmax turns matching scores into weights, which form a weighted sum of V. A larger weight need not produce a larger output contribution: V has magnitude and sign too.",
        ),
    ),
    (
        ("图解与操作", "Diagram and controls"),
        (
            "用查询旁的 ‹ › 选择位置，点击 token 卡片观察对应的 score、概率、V 与 p×V。卡片角标蓝色表示历史缓存，金色表示本轮计算；高亮强调所选位置，不可见的未来位置权重为零。“内容、贡献、汇聚结果”依次显示所选 V、它的加权贡献和完整 head 输出。向量条共用尺度；每张卡片的数字是第一个通道的值。点击头按钮或在子地图按 N 切换 head。长前缀用卡片上方的 ‹ › 翻页，输出仍包含所有位置。",
            "Use ‹ › beside Query to choose a position. Click a token card to inspect its score, probability, V and p×V. Blue corner markers identify cached history and gold markers identify new computation; highlighting identifies selection; masked future positions have zero weight. Content, Contribution and Result show the selected V, its weighted contribution and the complete head output. Strips share one scale; each number is the first channel. Click the head button or press N in the child room to inspect another head. Page through long prefixes using the upper ‹ ›; output still includes every position.",
        ),
    ),
    (
        ("符号与公式", "Symbols and formulas"),
        (
            "t 为 query 位置，j 为 key 位置，h 为查询头，g(h) 为它读取的键值头。q、k、v 的通道数等于 head 维度；softmax 沿 key 位置计算，可见位置的概率和为 1。图中的匹配分数尚未经过屏蔽，未来位置在 softmax 前设为 −∞。点积除以 head 维度的平方根，可缓和分数随维度增长的尺度。o 是一个头的汇聚结果；H 个头拼接后，经 W_o 投影得到 y。",
            "t is the query position, j the key position, h the query head and g(h) its key/value head. q, k and v have the width of one head. Softmax runs across key positions; visible probabilities sum to one. Displayed match scores are unmasked; future scores become −∞ before softmax. Dividing the dot product by the square root of head width moderates its scale. o is one head's result; concatenating H heads and projecting through W_o produces y.",
        ),
    ),
    (
        ("一个小例子", "A small example"),
        (
            "若两个可见位置的权重为 0.25、0.75，V 分别为 [2, −1]、[0, 3]，贡献是 [0.5, −0.25] 与 [0, 2.25]，输出为 [0.5, 2]。加权求和沿位置进行，保留 head 的特征维度；并不是挑一个 V，也不是把概率当作输出向量。",
            "For weights 0.25 and 0.75 with V=[2, −1] and [0, 3], contributions are [0.5, −0.25] and [0, 2.25]; output is [0.5, 2]. The sum runs across positions and preserves feature width. It neither selects a single V nor uses the probabilities themselves as the output vector.",
        ),
    ),
    (
        ("多头与 GQA", "Heads and GQA"),
        (
            "当前配置有 {heads} 个 Q head、{kv} 个 K/V head，每个 head {head_dim} 维。每个 Q head 独立计算 score 和输出；GQA 只共享 K/V，不合并 query。所有 Q head 输出拼接一次，再经 W_o 投影回 {dim} 维，之后进入残差连接。注意力权重描述本次匹配，不等同于语义解释或因果重要性。",
            "This configuration has {heads} Q heads, {kv} K/V heads and {head_dim} channels per head. Every Q head computes its own scores and output. GQA shares K/V, not query outputs. All Q outputs are concatenated once and projected through W_o back to {dim} channels before the residual connection. Attention weights describe these matches; they are not a direct semantic explanation or measure of causal importance.",
        ),
    ),
    (
        ("缓存与因果性", "Cache and causality"),
        (
            "每层复用旋转后的历史 K 和未旋转的 V，不缓存 Q，也不重复旋转旧 K。单 token 解码的 query 位于有效 key 的末尾，因此没有未来 key 可屏蔽；预填充的较早 query 或一次追加多个 token 时仍必须屏蔽未来位置。缓存不取消因果约束。若当前 trace 只有一个 query，‹ › 不会凭空生成历史 query。",
            "Each layer reuses rotated historical K and unrotated V. Q is not cached, and old K is not rotated again. A single-token decoding query is at the end of valid keys, so no future key exists to mask. Earlier prefill queries and multi-token chunks still require masking. Cache does not remove causality. If a trace contains only one query, ‹ › cannot create historical query rows.",
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
