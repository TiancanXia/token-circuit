"""Read-only bilingual model documents. Add topics here, not in window handlers."""

from dataclasses import dataclass
from ..i18n import Locale


@dataclass(frozen=True)
class Topic:
    title: str
    role: tuple[str, str]
    calculation: str
    shape: tuple[str, str]
    notes: tuple[str, str]


@dataclass(frozen=True)
class Document:
    title: str
    sections: tuple[tuple[str, str], ...]


# Row-vector notation; matrices in this demo are stored output × input.
TOPICS = {
    "overview": Topic(
        "Transformer",
        (
            "一个仅解码器网络：用已知前缀预测下一 token。小人代表前缀末尾的 token，各模块显示它经过的计算。",
            "A decoder-only network predicts the next token from a known prefix. Your character represents the final token of that prefix.",
        ),
        "Embedding → [RMSNorm → Attention → Residual → RMSNorm → FFN/MoE → Residual] × blocks\n→ Final RMSNorm → Unembedding → Softmax → Next token",
        (
            "前缀 token ID → {dim} 维特征 → {vocab} 个词表分数。每个 block 保持模型维度。",
            "Prefix token IDs → {dim} feature channels → {vocab} vocabulary scores. Every block preserves the model width.",
        ),
        (
            "Attention 汇聚不同位置的信息，FFN 独立变换各位置的特征。每个 block 的两个分支都通过残差连接汇入主流。",
            "Attention gathers information across positions; the FFN transforms each position independently. Residual connections join both branches back into the main stream.",
        ),
    ),
    "embedding": Topic(
        "Embedding",
        (
            "每个 token ID 从 embedding 表中取出自己的特征向量。",
            "Look up a feature vector for each token ID in the embedding table.",
        ),
        "X[t] = E[token_id[t]]",
        (
            "E: {vocab} × {dim}；每个 token 得到 {dim} 维向量。",
            "E: {vocab} × {dim}; each token receives {dim} channels.",
        ),
        (
            "这里不是 one-hot 向量；也没有把位置编码直接加到 embedding 上，位置信息稍后由 RoPE 进入 Q/K。",
            "The result is not a one-hot vector. Position information is introduced later through RoPE on Q/K, rather than added to this embedding.",
        ),
    ),
    "norm": Topic(
        "RMSNorm",
        (
            "按特征的均方根缩放每个 token。两个分支使用 pre-norm，输出前还有一次最终归一化。",
            "Scale each token by its root mean square. Attention and FFN use pre-norm; a final normalization precedes vocabulary projection.",
        ),
        "RMSNorm(x) = γ ⊙ x / sqrt(mean(x²) + ε)",
        (
            "{dim} → {dim} 维；沿特征维计算，不混合 token 位置。",
            "{dim} → {dim} channels; normalize features without mixing token positions.",
        ),
        (
            "不减去均值，区别于 LayerNorm。γ 为通道缩放系数，这里 γ=1、ε=10⁻⁶；残差旁路保留归一化前的流。",
            "Unlike LayerNorm, RMSNorm does not subtract the mean. γ scales channels; here γ=1 and ε=10⁻⁶. Residual paths preserve the stream before normalization.",
        ),
    ),
    "qkv": Topic(
        "Q / K / V",
        (
            "同一份归一化输入分别投影为查询 Q、匹配线索 K 和内容 V。先投影，再按 head 拆分。",
            "Project the same normalized input into queries Q, matching features K and content V, then split channels into heads.",
        ),
        "Q = X Wqᵀ\nK = X Wkᵀ\nV = X Wvᵀ",
        (
            "Wq: {dim} × {dim}；Wk/Wv: {kv_dim} × {dim}。{heads} 个 Q head，{kv} 个 KV head，每个 head 的维度 d_head = {head_dim}。",
            "Wq: {dim} × {dim}; Wk/Wv: {kv_dim} × {dim}. {heads} Q heads and {kv} KV heads, each with width d_head = {head_dim}.",
        ),
        (
            "MHA 每个 Q head 有独立 K/V；GQA 组内共享 K/V；MQA 所有 Q 共用一组 K/V。Q head 始终独立计算，K 和 V 权重也独立。",
            "MHA gives each Q head its own K/V. GQA shares K/V within groups; MQA shares one K/V head across all queries. Query heads remain independent, as do K and V projection weights.",
        ),
    ),
    "rope": Topic(
        "RoPE / KV cache",
        (
            "位置决定 Q/K 每对坐标的旋转角度，点积因此携带相对位置信息。缓存复用该层已计算的历史 K/V。",
            "Position determines the rotation of each Q/K coordinate pair, giving dot products relative position information. The cache reuses this layer’s historical K/V.",
        ),
        "θ_i = position / 10000^(2i / d_head)\n[x′, y′] = [x cos θ − y sin θ, x sin θ + y cos θ]\nK_all = [cached K_rotated; new K_rotated]\nV_all = [cached V; new V]",
        (
            "Q/K head: {head_dim} 维。每层缓存量 = 2 × 前缀长度 × {kv_dim} 个标量。",
            "Q/K heads: {head_dim} channels. Cache per layer = 2 × prefix length × {kv_dim} scalars.",
        ),
        (
            "只旋转 Q/K，不旋转 V。历史 K 已旋转，不能再次旋转；不缓存 Q。预填充计算全前缀，后续逐字解码只计算新 token。使用前缀绝对位置；上下文改变后需要相应更新缓存。",
            "Rotate Q/K only, never V. Cached K is already rotated; do not rotate it again. Q is not cached. Prefill computes the full prefix; subsequent single-token decode computes only the new token at its absolute position. Changes to the context require corresponding cache updates.",
        ),
    ),
    "attention": Topic(
        "Scaled Dot-Product Attention",
        (
            "每个 Q head 在可读位置中寻找匹配，再用概率汇聚对应的 V。多头并行，最后拼接一次。",
            "Each query head finds matches across visible positions and gathers its corresponding V using probability weights. Heads run independently and are packed once.",
        ),
        "scores = Q Kᵀ / sqrt(d_head)\nscores[j > query_position] = −∞  (only if future keys exist)\np = softmax(scores)\nhead_output = Σ_j p[j] V[j]\npacked = Concat(head_1, …, head_H)",
        (
            "{heads} 个 Q head × {head_dim} 维 → 拼接为 {dim} 维。Softmax 沿 key 位置归一化，不沿特征通道。",
            "{heads} Q heads × {head_dim} channels → {dim} packed channels. Softmax normalizes key positions, not feature channels.",
        ),
        (
            "单 token 解码若只含历史与当前有效 key，没有未来位置，可以省略 causal mask。多 token 预填充或 chunk 分块解码仍须屏蔽未来；缓存开关不是判断依据。MHA/GQA 是整个 Attention 的配置，不是额外的一层。",
            "Single-token decode with only valid past/current keys has no future positions, so an explicit causal mask can be omitted. Multi-token prefill or chunk decode must still hide future keys; the cache toggle alone cannot decide this. MHA/GQA configure the whole Attention layer rather than adding an operation.",
        ),
    ),
    "project": Topic(
        "Output · W_o",
        (
            "对已经拼接好的各 head 输出做线性混合，形成 Attention 分支的结果。",
            "Linearly mix the already-packed head outputs into the Attention branch result.",
        ),
        "Y = packed W_oᵀ",
        (
            "W_o: {dim} × {dim}；{dim} → {dim} 维。",
            "W_o: {dim} × {dim}; {dim} → {dim} channels.",
        ),
        (
            "Concat 在各头的 Scaled Dot-Product Attention 之后完成。所选 head 使用 W_o 对应的列块计算贡献；各 head 的贡献相加，得到完整输出。下一步把结果加回 RMSNorm 前的残差流。",
            "Concat follows the per-head Scaled Dot-Product Attention. Each head contributes through its corresponding columns of W_o; summing these contributions gives the full output. The next stage adds it to the stream preserved before RMSNorm.",
        ),
    ),
    "residual": Topic(
        "Residual +",
        (
            "让分支处理过的特征与进入该分支前的原始流相加。每个 block 有 Attention 和 FFN 两处残差。",
            "Add processed branch features to the stream preserved before that branch. Each block has Attention and FFN residual paths.",
        ),
        "x_next = x_before_norm + branch(RMSNorm(x_before_norm))",
        (
            "两个 {dim} 维向量逐通道相加，结果仍为 {dim} 维。",
            "Add two {dim}-channel vectors element by element; the result remains {dim} channels.",
        ),
        (
            "旁路保留归一化前的流，不是归一化后的向量。第二处残差结束后，该 block 完成。",
            "The bypass preserves the pre-normalization stream, not the normalized vector. The second residual completes the block.",
        ),
    ),
    "up": Topic(
        "FFN · Up",
        (
            "扩展每个 token 的特征通道，让后续非线性有更多组合空间。",
            "Expand each token’s feature channels before the nonlinearity.",
        ),
        "h = x W_upᵀ",
        (
            "W_up: {hidden} × {dim}；d_model = {dim} → d_ff = {hidden}。",
            "W_up: {hidden} × {dim}; d_model = {dim} → d_ff = {hidden}.",
        ),
        (
            "线性投影，不是图片上采样；不混合不同 token。完整 FFN 是 Up → GELU → Down。",
            "This is a linear projection, not image upsampling, and does not mix tokens. The complete FFN is Up → GELU → Down.",
        ),
    ),
    "activation": Topic(
        "GELU",
        (
            "让每个特征独立经过非线性，使 FFN 超出单个线性映射的表达能力。",
            "Apply a nonlinearity independently to each feature so the FFN is more expressive than a single linear map.",
        ),
        "GELU(x) ≈ 0.5 x [1 + tanh(sqrt(2/π) (x + 0.044715 x³))]",
        (
            "d_ff = {hidden}，逐通道计算，维度不变。",
            "d_ff = {hidden}; apply element by element without changing width.",
        ),
        (
            "使用常见 tanh 近似。它不是二值开关，也不会把所有负值直接变成零。",
            "A common tanh approximation is shown above. GELU is not a binary switch and does not zero every negative input.",
        ),
    ),
    "down": Topic(
        "FFN · Down",
        (
            "将扩展、激活后的特征映射回模型维度，准备进行残差相加。",
            "Project expanded, activated features back to model width before the residual addition.",
        ),
        "h = GELU(x W_upᵀ)\ny = h W_downᵀ",
        (
            "W_down: {dim} × {hidden}；d_ff = {hidden} → d_model = {dim}。",
            "W_down: {dim} × {hidden}; d_ff = {hidden} → d_model = {dim}.",
        ),
        (
            "Up 和 Down 的权重独立，不互为逆矩阵。",
            "Up and Down have independent weights; they are not inverse matrices.",
        ),
    ),
    "router": Topic(
        "MoE · Router",
        (
            "按当前 token 的特征选出少数专家，并计算所选专家的混合权重。",
            "Choose a few experts from the current token’s features and compute their mixing weights.",
        ),
        "scores = x W_routerᵀ\nselected = top_k(scores)\nweights = softmax(scores[selected])",
        (
            "{experts} 个专家，选 {top_k} 个；路由输入为 {dim} 维。",
            "{experts} experts; select {top_k}. Router input has {dim} channels.",
        ),
        (
            "每个 token 单独路由，仅对选中的专家分数归一化，得到混合权重。",
            "Route each token independently. Normalize the selected expert scores to obtain mixing weights.",
        ),
    ),
    "experts": Topic(
        "Expert FFNs",
        (
            "选中的专家并行运行自己的完整 FFN，未选中的专家不贡献输出。",
            "Selected experts run their own complete FFNs in parallel; idle experts do not contribute.",
        ),
        "expert_e(x) = GELU(x W_up,eᵀ) W_down,eᵀ",
        (
            "每个专家：d_model → d_ff → d_model（{dim} → {hidden} → {dim}），拥有独立权重。",
            "Each expert: d_model → d_ff → d_model ({dim} → {hidden} → {dim}), with independent weights.",
        ),
        (
            "专家是并行分支，各自拥有独立权重，可以对同一个输入产生不同的特征变换。",
            "Experts are parallel branches with independent weights; each can transform the same input differently.",
        ),
    ),
    "combine": Topic(
        "MoE · Mix",
        (
            "用 Router 给出的概率加权合并所选专家的输出。",
            "Combine selected expert outputs using the Router weights.",
        ),
        "y = Σ_e weight[e] × expert_e(x)",
        (
            "{top_k} 个 {dim} 维专家输出 → 一个 {dim} 维向量。",
            "{top_k} selected {dim}-channel expert outputs → one {dim}-channel vector.",
        ),
        (
            "不是简单平均，也不是把各专家向量拼接起来；之后仍须残差相加。",
            "This is neither an unweighted average nor a concatenation. A residual addition follows.",
        ),
    ),
    "unembedding": Topic(
        "Unembedding",
        (
            "把模型特征映射为整个词表的下一 token 分数。",
            "Map model features to next-token scores across the whole vocabulary.",
        ),
        "logits = x W_vocabᵀ",
        (
            "W_vocab: {vocab} × {dim}；{dim} 个特征 → {vocab} 个 logits。",
            "W_vocab: {vocab} × {dim}; {dim} features → {vocab} logits.",
        ),
        (
            "logits 可正可负，还不是概率。词表大小与特征维度不同；预测的是下一个 token，不是还原输入。",
            "Logits may be positive or negative and are not yet probabilities. Vocabulary size differs from feature width. This predicts the next token rather than reconstructing the input.",
        ),
    ),
    "softmax": Topic(
        "Softmax",
        (
            "将全词表 logits 转为总和为 1 的概率分布。",
            "Convert full-vocabulary logits into probabilities summing to one.",
        ),
        "p_i = exp(logit_i − max(logits)) / Σ_j exp(logit_j − max(logits))",
        (
            "{vocab} 个 logits → {vocab} 个概率；温度为 1。",
            "{vocab} logits → {vocab} probabilities; temperature is 1.",
        ),
        (
            "Top 7 列出概率最高的候选，保留全词表原概率，其余候选的概率合计单独列出。",
            "Top 7 lists the most likely candidates with their original full-vocabulary probabilities. The total probability of other candidates is listed separately.",
        ),
    ),
    "sample": Topic(
        "Next token / Sampling",
        (
            "决定下一 token，再从 Embedding 开始下一轮自回归计算。",
            "Choose the next token and start the next autoregressive round at Embedding.",
        ),
        "prefix → forward → next token → append to prefix → forward …",
        (
            "从 {vocab} 个候选中选择一个 token；前缀只追加一次。",
            "Choose one token from {vocab} candidates; append it to the prefix once.",
        ),
        (
            "随机抽样可以选中概率较低的 token；贪心选择则始终取概率最大的候选。两者是不同的选择方式。",
            "Random sampling from the full vocabulary can choose a less likely token. Greedy selection always takes the most likely candidate. These are different ways to choose.",
        ),
    ),
}

ALIASES = {
    **dict.fromkeys(("norm_attn", "norm_ffn", "norm_final"), "norm"),
    **dict.fromkeys(("residual_attn", "residual_ffn"), "residual"),
}


def topic_for(kind):
    """Map execution stage variants to their shared teaching topic."""
    topic = ALIASES.get(kind, kind)
    if topic not in TOPICS:
        raise ValueError(f"No guide topic for {kind}")
    return topic


def document(topic, config, language, guided, cache_enabled, phase):
    """Build a configuration snapshot; no tensors are evaluated or state mutated."""
    locale = Locale(language)
    item = TOPICS[topic]
    index = language == "en"
    context = locale.text(
        "guide.context",
        dim=config.dim,
        hidden=config.hidden,
        blocks=len(config.blocks),
        heads=config.heads,
        kv=config.num_kv_heads,
        head_dim=config.head_dim,
        mode=config.attention_mode,
        vocab=len(config.vocab),
    )
    context += "\n" + locale.text("phase." + phase)
    context += " · KV cache " + locale.text("on" if cache_enabled else "off")
    shape = item.shape[index].format(
        dim=config.dim,
        hidden=config.hidden,
        vocab=len(config.vocab),
        heads=config.heads,
        kv=config.num_kv_heads,
        head_dim=config.head_dim,
        kv_dim=config.kv_dim,
        experts=config.experts,
        top_k=config.top_k,
    )
    notes = item.notes[index]
    if topic in ("overview", "softmax", "sample"):
        notes += "\n" + locale.text("guide.values")
    if guided and topic in ("softmax", "sample"):
        notes += "\n" + locale.text("journey_probs")
    if topic == "attention":
        from .attention_guide import sections

        return Document(
            item.title,
            sections(config, language)
            + (
                (
                    locale.text("guide.section.context"),
                    context + "\n" + locale.text("guide.values"),
                ),
            ),
        )
    values = (context, item.role[index], item.calculation, shape, notes)
    keys = ("context", "role", "calculation", "shape", "notes")
    return Document(
        item.title,
        tuple(
            (locale.text("guide.section." + key), value)
            for key, value in zip(keys, values)
        ),
    )
