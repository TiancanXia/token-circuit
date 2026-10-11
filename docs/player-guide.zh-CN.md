# 玩家指南

[English](player-guide.md) | [简体中文](player-guide.zh-CN.md) · [返回首页](../README.zh-CN.md)

## 第一次旅程

1. 启动游戏，最下面只亮起句子的第一个 token。
2. 用方向键或 WASD 靠近 Embedding，按 E / Enter 进入。
3. 在房间中靠近装置，按 E 处理；完成后从亮起的出口离开。
4. Attention 内还可以进入 QKV、RoPE、Head attention、Output 的内部地图。M 返回上一层，进度保留。
5. 完成输出模块后，小人消散并在模型开头聚拢，句子亮起下一个 token；重复直到句子结束。

F 会快速走完当前的一次 forward。G 打开定位面板：左右选择 block 或输入/输出区，上下选择模块，Enter 定位；定位不会提前点亮下一个 token。

## 操作速查

| 按键 | 功能 |
| --- | --- |
| 方向键 / WASD | 移动 |
| E / Enter | 进入、处理、离开 |
| M / Esc | 返回上一层；Esc 先退出全屏，主地图退出 |
| F | 快速完成当前一轮 |
| G | 定位面板；再次 G 或 Esc 关闭 |
| H | MHA / GQA 切换，重新开始 |
| C | KV cache 开关，重新开始 |
| N / Tab | 切换观察的 Q head / 已到达的层 |
| L / R / F11 | 中英文切换 / 重新开始 / 全屏 |

点击说明按钮阅读模块文档。文档打开时游戏暂停；Esc 只关闭文档。语言切换会重新开始，保留当前注意力和缓存设置。

## Attention 观察台

在 Head attention / Softmax 点击 token 卡片查看贡献。查询旁的箭头选择已有 query，Q→KV 或 N 切换 head；长前缀可翻页。被屏蔽的未来位置贡献为零。这些操作只改变观察，不推进计算。说明按钮提供公式与算例。

## 常见问题

**为什么有黑色格子？** 后续 token 尚未亮起，每完成一整轮才出现一个。

**下一个字为什么不是概率最高的字？** 旅程使用固定句子，文字与概率图独立；图里是未训练微型模型的计算结果。Top 7 保留原来的全词表概率，不会重新归一化。

**无法启动？** 需要 Python 3.10+、Tkinter 和图形桌面。先运行 `python -m tkinter` 检查 Tk。Windows 启动失败时窗口会保留错误信息，可附到问题反馈中。

**怎样反馈？** 在 [Issues](https://github.com/TiancanXia/token-circuit/issues) 描述系统、Python 版本、操作步骤与报错；截图能帮助定位问题。
