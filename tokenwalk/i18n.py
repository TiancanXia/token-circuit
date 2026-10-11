"""Shared UI copy. Each entry is (Chinese, English); no game logic here."""

from dataclasses import dataclass
from .model.architecture import CONTENT


TEXTS = {
    "step.qkv.0": ("读取输入", "Read input"),
    "step.qkv.1": ("生成查询", "Build Q"),
    "step.qkv.2": ("生成键", "Build K"),
    "step.qkv.3": ("生成值", "Build V"),
    "step.qkv.4": ("多头布局", "Head layout"),
    "step.rope.0": ("读取位置", "Position"),
    "step.rope.1": ("旋转 Q/K", "Rotate Q/K"),
    "step.rope.2": ("准备键值", "Ready K/V"),
    "step.project.0": ("拼接输入", "Packed input"),
    "step.project.1": ("混合通道", "Mix channels"),
    "step.project.2": ("注意力输出", "Output"),
    "step.attention.0": ("读取内容", "Read"),
    "step.attention.1": ("计算匹配", "Match"),
    "step.attention.2": ("屏蔽未来", "Mask"),
    "step.attention.visible": ("可读位置", "Read keys"),
    "step.attention.3": ("分配权重", "Weight"),
    "step.attention.4": ("汇聚内容", "Gather"),
    "step.attention.5": ("拼接各头", "Pack heads"),
    "probe.content": ("内容", "Content"),
    "probe.part": ("贡献", "Contribution"),
    "probe.total": ("汇聚结果", "Result"),
    "probe.head_hint": ("N · 头 {head}/{count}", "N · Head {head}/{count}"),
    "probe.head": ("头 {head} · 键值组 {kv}", "Head {head} · KV group {kv}"),
    "probe.match": ("匹配分数", "Match scores"),
    "probe.weights": ("位置权重", "Position weights"),
    "probe.query": ("查询 {position} · {token}", "Query {position} · {token}"),
    "probe.pick": ("点击 token 查看贡献", "Click a token to inspect"),
    "probe.masked": ("不可见", "Masked"),
    "probe.cached": ("历史缓存", "Cached"),
    "probe.new": ("本轮计算", "Computed"),
    "probe.key": (
        "位置 {index} · 匹配 {score:+.3f} · {status}",
        "Position {index} · match {score:+.3f} · {status}",
    ),
    "guide.values": (
        "这里的模型尚未训练，数值用于观察计算过程。",
        "This model has not been trained; its values illustrate the calculation.",
    ),
    "journey_probs": (
        "旅程文字与概率图独立。",
        "The journey text is independent of this probability chart.",
    ),
    "guide.button": ("模型说明", "Model guide"),
    "expert_crop": (
        "显示 {shown}/{total} 专家 · 省略权重 {rest:.1%}",
        "{shown}/{total} experts shown · omitted weight {rest:.1%}",
    ),
    "expert_crop_wait": (
        "显示 {shown}/{total} 专家 · 等待路由",
        "{shown}/{total} experts shown · waiting for routing",
    ),
    "guide.title": ("模型手册", "Model handbook"),
    "guide.close": ("关闭 · Esc", "Close · Esc"),
    "guide.footer": (
        "阅读时暂停 · 滚轮翻阅 · 关闭后继续",
        "Paused while reading · scroll to explore · close to continue",
    ),
    "guide.context": (
        "{blocks} 层 · {mode} {heads}Q/{kv}KV\n模型维度 {dim} · FFN {hidden} · 词表 {vocab}",
        "{blocks} blocks · {mode} {heads}Q/{kv}KV\nModel width {dim} · FFN {hidden} · vocabulary {vocab}",
    ),
    "guide.section.context": ("模型参数", "Model parameters"),
    "guide.section.role": ("它做什么", "What it does"),
    "guide.section.calculation": ("如何计算", "Calculation"),
    "guide.section.shape": ("输入与输出", "Input and output"),
    "guide.section.notes": ("理解要点", "Things to understand"),
    "brief.qkv.0": ("一份输入，三组投影", "One input, three projections"),
    "brief.qkv.1": ("Q：提出查询", "Q: ask a query"),
    "brief.qkv.2": ("K：提供匹配线索", "K: matching features"),
    "brief.qkv.3": ("V：携带内容", "V: carry content"),
    "brief.qkv.4": ("Q 头与 K/V 头的对应关系", "Query-to-K/V head mapping"),
    "brief.rope.0": ("位置决定旋转角度", "Position sets the rotation"),
    "brief.rope.1": ("旋转 Q/K，V 不变", "Rotate Q/K; leave V unchanged"),
    "brief.rope.ready.prefill": (
        "准备当前位置的 K/V",
        "Prepare K/V at these positions",
    ),
    "brief.rope.ready.decode": (
        "复用历史，接入新 K/V",
        "Reuse history; append new K/V",
    ),
    "brief.rope.ready.full": ("准备全前缀 K/V", "Prepare full-prefix K/V"),
    "brief.attention.read.prefill": (
        "预填充：计算全前缀",
        "Prefill: compute the full prefix",
    ),
    "brief.attention.read.decode": (
        "历史 K/V + 当前 K/V",
        "Historical K/V + current K/V",
    ),
    "brief.attention.read.full": ("本轮重算全前缀", "Recompute the full prefix"),
    "brief.attention.visible": (
        "没有未来 key，直接放行",
        "No future keys; pass through",
    ),
    "brief.attention.1": ("查询与各位置匹配", "Match the query to each key"),
    "brief.attention.2": (
        "未来位置的概率为零",
        "Future positions receive zero probability",
    ),
    "brief.attention.3": ("分数转为概率", "Turn scores into probabilities"),
    "brief.attention.4": ("按概率汇聚 V", "Gather V using probabilities"),
    "brief.attention.5": ("各头输出只拼接一次", "Pack head outputs once"),
    "brief.project.0": ("查看已拼接的输入", "Inspect the packed input"),
    "brief.project.1": ("混合特征通道", "Mix feature channels"),
    "brief.project.2": ("下一站：残差相加", "Next: residual addition"),
    "head_groups": (
        "{mode} · {q}Q / {kv}KV · 展示前 {shown} 个 Q head",
        "{mode} · {q}Q / {kv}KV · showing first {shown} Q heads",
    ),
    "cache_stats": (
        "{phase} · 复用 {old} / 计算 {new} · 末 {shown} 个位置",
        "{phase} · reused {old} / computed {new} · last {shown} positions",
    ),
    "phase.prefill": ("预填充 / 全前缀", "Prefill / full prefix"),
    "phase.decode": ("单 token 解码", "Single-token decode"),
    "phase.full": ("全前缀重算", "Full-prefix recompute"),
    "mask_bypass": (
        "当前查询只读取已有位置",
        "This query reads the existing positions",
    ),
    "mask_axes": (
        "query 行 × key 列 · 末 {rows} 行 / {keys} 列",
        "Query rows × key columns · last {rows} rows / {keys} columns",
    ),
    "concat_shape": (
        "{q} 个 Q head × {d} 维 = {total} 维 · 展示前 {shown} 个 head",
        "{q} Q heads × {d} channels = {total} channels · first {shown} heads shown",
    ),
    "concat_order": (
        "按 head 顺序拼接；GQA 共用 K/V，保留全部 Q head 的输出。",
        "Pack in head order; GQA shares K/V while retaining every Q head output.",
    ),
    "packed_input": (
        "已拼接的输入：保留每个 Q head 的输出，供 W_o 混合特征。",
        "Already packed: every Q head output is retained for W_o to mix.",
    ),
    "await_attention_stage": (
        "当前节点 {title} 尚未处理 · E 深入",
        "{title} has not run yet · E to explore",
    ),
    "attention_layout": (
        "MHA 独立 K/V · GQA 组内共享",
        "MHA: separate K/V · GQA: shared K/V",
    ),
    "cache_size": (
        "该层 K/V 共 {n} 个标量 · 相对 MHA 少 {saved} 个",
        "Layer K/V: {n} scalars · {saved} fewer than MHA",
    ),
    "cache_off_note": ("缓存关闭 · 重算全部 K/V", "Cache OFF · recompute all K/V"),
    "rope_absolute": ("沿用前缀绝对位置", "Keep the absolute prefix position"),
    "attention_positions": (
        "key {start}–{end} · query {pos}",
        "Keys {start}–{end} · query {pos}",
    ),
    "mask_last": (
        "越过当前 query 的 key 被遮住",
        "Hide keys beyond each query position",
    ),
    "project_residual": (
        "下一站：与 RMSNorm 前保留的残差流相加。",
        "Next: add the preserved stream from before RMSNorm.",
    ),
    "detail_hint": (
        "E 逐步处理 · M 返回上一层 · N 切换观察 head",
        "E process steps · M parent room · N inspect another head",
    ),
    "detail_order": ("先完成前一个节点", "Complete the preceding node first"),
    "detail_step": (
        "已展开 {title} · 前往下一节点",
        "{title} revealed · continue to the next node",
    ),
    "head_hint": (
        "观察 Q head {n}，计算不变",
        "Inspecting Q head {n}; computation unchanged",
    ),
    "detail_level": ("{house} · 第 {n} 层 · {title}", "{house} · Layer {n} · {title}"),
    "detail_enter": ("E 深入", "E explore"),
    "weighted_crop": (
        "显示贡献 {shown:+.4f} + 省略贡献 {earlier:+.4f} = {total:+.4f}",
        "Shown {shown:+.4f} + omitted {earlier:+.4f} = {total:+.4f}",
    ),
    "attention_mode": ("H · {mode}", "H · {mode}"),
    "cache_mode": ("C · KV cache {state}", "C · KV cache {state}"),
    "on": ("开", "ON"),
    "off": ("关", "OFF"),
    "window": ("Token 回路 · 逐字探索", "Token Circuit · A Walk Through Thought"),
    "title": ("Token 回路", "Token Circuit"),
    "language": ("L / EN", "L / ZH"),
    "language_hint": ("L 切换语言并重来", "L changes language & restarts"),
    "layer": ("第 {n} 层", "Layer {n}"),
    "round": ("第 {n} 轮", "Round {n}"),
    "progress": ("{n} / {total} 字", "{n} / {total} tokens"),
    "inside": ("小地图 / {title}", "Inside / {title}"),
    "macro": ("大地图 / Transformer", "Main map / Transformer"),
    "flight": ("F / 快速 forward", "F / Fast forward"),
    "birth": ("下一 token / output → input", "Next token / output → input"),
    "complete": ("句子完成", "Sequence complete"),
    "paused": ("暂停", "Paused"),
    "keys": (
        "方向键 行走 · E 交互 · F forward · G 定位 · M 返回 · R 重来",
        "Arrows move · E interact · F forward · G go to · M back · R restart",
    ),
    "window_keys": ("F11 全屏 · ESC 返回 / 退出", "F11 fullscreen · ESC back / quit"),
    "block": ("当前 {n}/{total} · {style}", "Layer {n}/{total} · {style}"),
    "enter": ("E 进入", "E enter"),
    "process": ("E 处理", "E process"),
    "processed": ("✓ 完成", "✓ done"),
    "exit": ("出口 / E", "Exit / E"),
    "exit_wait": ("出口未亮", "Exit pending"),
    "process_exit": ("E 处理 / 出口", "E process / exit"),
    "restart": ("R 重来", "R restart"),
    "fast_forward": ("F 快速 forward", "F fast forward"),
    "flying": ("飞行中 …", "Flying …"),
    "returning": ("返回入口 …", "Returning to input …"),
    "transferring": ("转场中 …", "Re-forming …"),
    "finished": ("已完成", "Finished"),
    "back_layers": ("M 返回 · Tab 切层", "M back · Tab layers"),
    "wait": ("等待 E", "Waiting for E"),
    "wait_compute": ("等待 E 计算", "E to compute"),
    "next_token": ("下一 token / {token}", "Next token / {token}"),
    "signs": ("金 + / 蓝 −", "Gold + / blue −"),
    "done_restart": ("句子走完了 · R 重来", "Sequence complete · R to restart"),
    "near_enter": ("靠近入口 · E 进入", "Approach a portal · E to enter"),
    "next_house": ("下一站 → {title}", "Next → {title}"),
    "room_hint": ("E 处理 · M 返回", "E process · M back"),
    "finish_first": ("先完成处理 · M 返回", "Finish this module first · M back"),
    "near_process": ("靠近节点 · E 处理", "Approach a node · E to process"),
    "already_done": ("已完成 · 前往下一节点", "Done · walk to the next node"),
    "process_first": ("先处理 → {title}", "Process first → {title}"),
    "step_done": ("{title} 完成 · {next}", "{title} done · {next}"),
    "leave_room": ("E 出口 / M 返回", "E exit / M back"),
    "next_node": ("前往下一节点", "Next node"),
    "revealing": ("下一 token → {token}", "Next token → {token}"),
    "fast_message": ("快速 forward …", "Fast forward …"),
    "tab_inside": ("Tab 在模块内切层", "Use Tab inside a module to change layers"),
    "only_layer": ("仅一层可查看", "Only one layer available"),
    "layer_message": ("第 {n} 层 · Tab 切层", "Layer {n} · Tab to change layers"),
    "new_token": ("「{token}」· E 进入 Embedding", "“{token}” · E to enter Embedding"),
    "project_hint": ("{label} / 按 E 投影", "{label} / E to project"),
    "not_computed": ("尚未计算", "Not computed"),
    "matrix_wait": ("矩阵尚未计算", "Matrix not computed"),
    "matrix_rows": (
        "末 {n} 行 · 金色行标 = 当前 token",
        "Last {n} rows · gold = current token",
    ),
    "rope_v": ("RoPE → Q / K · V 不变", "RoPE → Q / K · V unchanged"),
    "rope_angle": (
        "位置 {pos} · 每个 head：θᵢ = pos / 10000^(2i/d_head)",
        "Position {pos} · per head: θᵢ = pos / 10000^(2i/d_head)",
    ),
    "matrix_signs": (
        "金 + / 蓝 − · 最后一行 = 当前小人",
        "Gold + / blue − · last row = your token",
    ),
    "attention_crop": (
        "Q 末 {rows} 行 × K 末 {keys} 列 · 原概率",
        "Last {rows} Q rows × {keys} K columns · original weights",
    ),
    "attention_flow": (
        "匹配 → {visibility} → 权重 → 汇聚 → 拼接 → 投影",
        "Match → {visibility} → Weight → Gather → Pack → Project",
    ),
    "all_visible": ("已有位置", "Read positions"),
    "causal_visibility": ("屏蔽未来", "Mask future"),
    "prefix": (
        "末 {n} 个前缀 token · 最后一个 = 小人",
        "Last {n} prefix tokens · last = your character",
    ),
    "features": ("ID {id} → {n} 个特征通道", "ID {id} → {n} feature channels"),
    "router": (
        "Router / top-{k} · 专家并行处理",
        "Router / top-{k} · parallel experts",
    ),
    "weight": ("权重 {weight:.1%}", "Weight {weight:.1%}"),
    "sleeping": ("未选中 / 休眠", "Not selected / idle"),
    "before_norm": ("原始特征", "Input features"),
    "after_norm": ("RMSNorm 后", "After RMSNorm"),
    "norm_formula": ("x / √(mean(x²) + ε) · γ = 1", "x / √(mean(x²) + ε) · γ = 1"),
    "skip": ("原始残差", "Original stream"),
    "branch": ("分支输出", "Branch output"),
    "sum": ("相加后", "Their sum"),
    "vocab_hint": (
        "按 E，把特征映射到词表。",
        "E to project features into the vocabulary.",
    ),
    "full_sample": ("按全词表概率采样", "Sample from the full vocabulary"),
    "vocab_size": ("词表：{n} 个 token", "Vocabulary: {n} tokens"),
    "output_logits_title": (
        "下一 token 的 logits · 词表 {n}",
        "Next-token logits · vocab {n}",
    ),
    "output_probs_title": (
        "下一 token 的概率 · 词表 {n}",
        "Next-token probabilities · vocab {n}",
    ),
    "probability_hint": (
        "按 E，将全词表 logits 归一化为概率。",
        "E to normalize all vocabulary logits into probabilities.",
    ),
    "top_logits": (
        "Top {n}/{total} logits · 尚未归一化",
        "Top {n}/{total} logits · not normalized",
    ),
    "top_probabilities": (
        "Top {n}/{total} · 其余 token 合计 {rest:.1%}",
        "Top {n}/{total} · rest of vocabulary {rest:.1%}",
    ),
    "logits_note": (
        "logits → Softmax → 下一 token 的概率",
        "logits → Softmax → next-token probabilities",
    ),
    "navigation_button": ("G / 定位", "G / Go to"),
    "navigation_title": ("定位到计算步骤", "Go to a forward step"),
    "navigation_note": (
        "自动计算前置步骤，保持当前 token；到达所选模块后按 E 处理。",
        "Earlier steps are computed; your current token stays. Press E at the selected module.",
    ),
    "navigation_keys": (
        "← → 选 block · ↑ ↓ 选模块 · Enter / 点击定位 · G / Esc 关闭",
        "Left / right: block · Up / down: module · Enter / click: go · G / Esc: close",
    ),
    "global_stages": ("输入 / 输出", "Input / output"),
    "jumped": ("已定位 → {title} · E 处理", "At {title} · E to process"),
}

# Technical stage titles stay shared. Reuse the original Chinese descriptions.
_STAGE_EN = {
    "embedding": "A landing port for words",
    "norm_attn": "The pool before attention",
    "qkv": "Three observation mirrors",
    "rope": "A rotating atlas of position",
    "attention": "Per-head attention and packing",
    "project": "Output projection",
    "residual_attn": "Meeting your earlier self",
    "norm_ffn": "The pool before the FFN",
    "up": "A workshop for growing features",
    "activation": "A garden of nonlinearity",
    "down": "A workshop for gathering features",
    "router": "A fork toward the experts",
    "experts": "Parallel expert workshops",
    "combine": "A weighted confluence",
    "residual_ffn": "Meeting yourself again",
    "norm_final": "The final pool",
    "unembedding": "A sky of possible words",
    "softmax": "The balance of probability",
    "sample": "The next star",
}
TEXTS.update(
    {
        f"stage.{kind}": (content[1], _STAGE_EN[kind])
        for kind, content in CONTENT.items()
    }
)
_RESULT_ZH = {
    "token embedding": "token 的向量",
    "normalized stream": "归一化后的流",
    "Q (current token)": "Q（当前 token）",
    "rotated Q": "旋转后的 Q",
    "concatenated heads": "拼接后的 heads",
    "attention branch": "Attention 分支",
    "residual stream": "残差流",
    "expanded features": "扩展后的特征",
    "GELU features": "GELU 后的特征",
    "FFN branch": "FFN 分支",
    "router input": "Router 输入",
    "expert inputs": "专家输入",
    "weighted expert output": "专家加权输出",
    "next-token logits": "下一 token 的 logits",
    "next-token probabilities": "下一 token 的概率",
    "sample distribution": "采样分布",
}
TEXTS.update({f"result.{label}": (zh, label) for label, zh in _RESULT_ZH.items()})


@dataclass(frozen=True)
class Locale:
    language: str = "zh"

    def __post_init__(self):
        if self.language not in ("zh", "en"):
            raise ValueError("Supported languages: zh, en")

    def text(self, key, **values):
        return TEXTS[key][self.language == "en"].format(**values)
