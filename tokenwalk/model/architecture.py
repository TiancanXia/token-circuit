"""Network content is declared here; window code does not construct its flow."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    dim: int = 8
    heads: int = 2
    hidden: int = 12
    experts: int = 3
    top_k: int = 2
    blocks: tuple = ("dense", "moe")
    vocab: tuple = ("风", "在", "停", "光", "树", "桥", "生长", "。")
    context: tuple = (0, 1, 2)
    kv_heads: int | None = None

    @property
    def num_kv_heads(self):
        return self.heads if self.kv_heads is None else self.kv_heads

    @property
    def head_dim(self):
        return self.dim // self.heads

    @property
    def kv_dim(self):
        return self.num_kv_heads * self.head_dim

    @property
    def attention_mode(self):
        return (
            "MHA"
            if self.num_kv_heads == self.heads
            else "MQA"
            if self.num_kv_heads == 1
            else "GQA"
        )

    def kv_head_for(self, query_head):
        if not 0 <= query_head < self.heads:
            raise ValueError("Invalid query head")
        return query_head // (self.heads // self.num_kv_heads)

    def __post_init__(self):
        if (
            self.dim <= 0
            or self.heads <= 0
            or self.dim % self.heads
            or (self.dim // self.heads) % 2
        ):
            raise ValueError("Use even head dimensions for RoPE")
        if not 1 <= self.num_kv_heads <= self.heads or self.heads % self.num_kv_heads:
            raise ValueError("Query heads must be divisible by KV heads")
        if self.hidden <= 0 or not 1 <= self.top_k <= self.experts:
            raise ValueError("Invalid FFN/expert dimensions")
        if not self.blocks or any(b not in ("dense", "moe") for b in self.blocks):
            raise ValueError("Unknown block type")
        if not self.context or any(i < 0 or i >= len(self.vocab) for i in self.context):
            raise ValueError("Invalid context")


@dataclass(frozen=True)
class Stage:
    key: str
    kind: str
    title: str
    subtitle: str
    explanation: str
    block: int = -1


CONTENT = {
    "embedding": (
        "Embedding",
        "词的登陆港",
        "从词表取出一行向量。小人代表上下文最后一个 token；身上的光点代表它的数值特征。",
    ),
    "norm_attn": (
        "RMSNorm",
        "Attention 前的整流池",
        "保存原始残差流，再按均方根缩放特征。RMSNorm 不减去均值；此处缩放参数为 1。",
    ),
    "qkv": (
        "Q / K / V",
        "三个观测镜",
        "同一份输入经过三组投影。Q 发出询问，K 提供匹配线索，V 携带稍后要聚合的内容。",
    ),
    "rope": (
        "RoPE",
        "位置的旋转星盘",
        "在每个 head 内，按 token 的位置旋转 Q 和 K 的成对坐标。V 保持原样；不是把位置加到 embedding 上。",
    ),
    "attention": (
        "SDPA · Concat",
        "逐头注意力与拼接",
        "各 Q head 独立计算 QKᵀ/√d_head，屏蔽实际存在的未来位置，softmax 后加权汇聚所对应的 V，最后按 head 顺序拼接。MHA/GQA 决定 Q head 与 K/V head 的对应关系，不是额外的一步。",
    ),
    "project": (
        "Output · W_o",
        "输出投影",
        "已拼接的各 head 输出经过 W_o 混合特征通道，回到残差流的维度。拼接已在逐头注意力的末尾完成。",
    ),
    "residual_attn": (
        "Residual +",
        "与原来的自己汇合",
        "将 Attention 分支的输出，加回进入 RMSNorm 前的原始向量。蓝色光索承载这一份旁路信号。",
    ),
    "norm_ffn": (
        "RMSNorm",
        "FFN 前的整流池",
        "保留新的残差流，再做一次 RMSNorm。接下来处理各个 token 的特征，而不是混合不同位置。",
    ),
    "up": (
        "FFN · Up",
        "特征的生长工坊",
        "上投影扩展特征维度。这是线性映射，不是空间插值；特征有了更多可组合的通道。",
    ),
    "activation": (
        "GELU",
        "非线性的花园",
        "每个特征独立经过 GELU。非线性使连续两次线性投影不再等同于一次线性映射。",
    ),
    "down": (
        "FFN · Down",
        "收束的工坊",
        "下投影把扩展后的特征送回模型维度，准备和旁路保留的向量相加。",
    ),
    "router": (
        "MoE · Router",
        "专家的分岔口",
        "路由器根据每个 token 的特征计算专家分数。选取 top-k 专家，并对所选分数归一化。",
    ),
    "experts": (
        "Expert FFNs",
        "并行的专家工坊",
        "被选专家分别完成上投影、GELU、下投影。画面显示当前 token 的选择；不同 token 可以选择不同专家。",
    ),
    "combine": (
        "MoE · Mix",
        "加权汇流港",
        "按路由权重合并所选专家的输出。专家是并行分支，不是让 token 先后经过所有专家。",
    ),
    "residual_ffn": (
        "Residual +",
        "第二次汇合",
        "将 FFN 或 MoE 的输出，加回进入第二个 RMSNorm 前的残差流。一个 block 到这里完成。",
    ),
    "norm_final": (
        "Final RMS",
        "离开网络前的整流池",
        "最后的残差流再经过一次 RMSNorm，然后映射到词表空间。",
    ),
    "unembedding": (
        "Unembedding",
        "词表的星空",
        "向量经过输出矩阵，得到每个词的 logit。这里预测下一个 token，不是把输入 token 原样解码回来。",
    ),
    "softmax": (
        "Softmax",
        "概率的天平",
        "对 logits 做 softmax，得到总和为 1 的概率分布。当前温度固定为 1。",
    ),
    "sample": (
        "Sampling",
        "下一颗星",
        "按概率进行一次随机抽样。最可能的词不一定被选中；重新出发会使用新的抽样种子。",
    ),
}


def build_stages(config):
    result = []

    def add(kind, block=-1):
        result.append(Stage(f"{block}:{kind}", kind, *CONTENT[kind], block))

    add("embedding")
    for block, ffn in enumerate(config.blocks):
        for kind in (
            "norm_attn",
            "qkv",
            "rope",
            "attention",
            "project",
            "residual_attn",
            "norm_ffn",
        ):
            add(kind, block)
        for kind in (
            ("up", "activation", "down")
            if ffn == "dense"
            else ("router", "experts", "combine")
        ):
            add(kind, block)
        add("residual_ffn", block)
    for kind in ("norm_final", "unembedding", "softmax", "sample"):
        add(kind)
    return result
