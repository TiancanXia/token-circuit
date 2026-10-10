# Token 回路

[English](README.md) | [简体中文](README.zh-CN.md)

控制像素小人穿过 Transformer，走进各个模块，观察一个 token 的计算过程。

![Transformer 大地图](assets/showcase/world.zh.png)

## 试玩

[下载 ZIP](https://github.com/TiancanXia/token-circuit/archive/refs/heads/main.zip) 并解压。安装带 Tkinter 的 [Python 3.10+](https://www.python.org/downloads/)。

Windows 双击 `start.cmd`，或运行：

```sh
python game.py
```

macOS / Linux 在图形桌面运行 `python3 game.py`，这两个平台尚未试玩验证。如果缺少 Tk，可用 `python3 -m tkinter` 检查，并安装对应平台的 Python Tk 支持。

靠近 Embedding 按 **E** 进入，按 **F** 可快速走完一轮计算。

![一轮计算](assets/showcase/forward.gif)

## 操作

| 操作 | 按键 |
| --- | --- |
| 移动 | 方向键 / WASD |
| 进入或处理 | E / Enter |
| 返回上一层地图 | M |
| 快速完成一轮 | F |
| 选择 block 与阶段 | G、方向键、Enter |
| 切换 MHA / GQA | H |
| 开关 KV cache | C |
| 查看 Q head / 已到达的层 | N / Tab |
| 语言 / 重来 / 全屏 | L / R / F11 |

H、C、L 会重新开始。Esc 根据当前界面关闭文档或定位面板、退出全屏、返回上一层，或在大地图退出游戏。

点击模块的说明按钮查看详细内容，阅读期间游戏暂停。

![QKV 投影](assets/showcase/attention.zh.png)

## 模型说明

游戏使用固定、未训练的参数计算一个小型 Transformer。句子与概率图独立，具体计算见 [模型说明](docs/model-audit.zh-CN.md)。

## 文档

- [玩家指南](docs/player-guide.zh-CN.md)
- [开发与贡献](CONTRIBUTING.zh-CN.md)
- [文档目录](docs/README.zh-CN.md)
- [更新日志](CHANGELOG.md) · [问题反馈](https://github.com/TiancanXia/token-circuit/issues)

## 许可

代码与原创视觉资源采用 [MIT](LICENSE)。句子引用 Alan Turing《Computing Machinery and Intelligence》(1950)，中文为译文：[出处](https://doi.org/10.1093/mind/LIX.236.433)。引用材料和系统字体保留各自权利。
