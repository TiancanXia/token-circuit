# 代码结构

[English](architecture.md) | [简体中文](architecture.zh-CN.md) · [首页](../README.zh-CN.md)

## 文件

| 位置 | 内容 |
| --- | --- |
| `game.py`、`start.cmd` | 启动入口 |
| `tokenwalk/app.py` | Tk 窗口、输入、焦点与计时 |
| `tokenwalk/i18n.py` | 中英文界面文字 |
| `tokenwalk/content/guide.py` | 模块说明 |
| `tokenwalk/model/architecture.py` | 配置与阶段 |
| `tokenwalk/model/mathops.py` | 数值算子 |
| `tokenwalk/model/forward.py` | 前向执行与结果快照 |
| `tokenwalk/model/cache.py` | 逐层 KV 缓存 |
| `tokenwalk/model/prompt.py` | 句子与词表 |
| `tokenwalk/simulation/` | 地图、房间、移动、飞行与转场 |
| `tokenwalk/view/` | Canvas 绘制、侧栏与文档窗口 |
| `tools/`、`tests/` | 预览与验证 |
| `assets/showcase/` | README 图片与动图 |

模型不依赖界面。模拟接收模型结果与时间步，视图绘制状态，`app.py` 处理窗口事件。

## 执行

手动 E 和快速 F 共用 `Session._execute_step`，`ForwardPass.execute` 检查阶段顺序。大地图只显示一个重复 block 框，但每个配置的 block 都实际执行。

Sample 后，小人从 Output 转移到 Embedding。1.4 秒动画结束后，句子和上下文只追加一次；处理句末 token 后结束。

`Session.jump_to(index)` 重建当前前缀的前置阶段，取消飞行和转场，打开目标阶段，不推进句子。G 使用真实阶段和 block 编号；N 改变观察的 Q head，Tab 选择已到达的层。

打开文档或失去焦点会暂停推进。文档关闭前也会阻止游戏操作。侧栏与文档通过 `view/context.py` 选择相同的阶段结果。

## 模块房间

`simulation/maps.py` 保存道路与模块分组；`simulation/rooms.py` 注册 `RoomSpec` 检查点，记录访问进度和父层位置。`view/module_views.py` 绘制模块房间，`view/detail_views.py` 绘制注册的子模块。

`ForwardPass.preview` 读取已完成结果，或对当前阶段进行脱离真实状态的预计算，不改变执行器、随机状态或缓存。子模块最后一个检查点只提交一次阶段。M 返回一级，F 先返回大地图路线再继续。

阶段诊断存于 `Result.details`。成对界面标签在 `i18n.py`，详细说明及别名在 `content/guide.py`。

### 添加内部地图

1. 在 `simulation/rooms.py` 中为算子注册 `RoomSpec` 检查点。
2. 将新增数值诊断存入该算子的 `Result.details`。
3. 在 `view/detail_views.py` 注册读取所选结果的绘制函数。
4. 在 `i18n.py` 添加双语标签，在 `content/guide.py` 添加主题或别名。

新地图复用公共地板、移动和输入处理。检查两种语言的进入、完成、返回与重复访问。

## 缓存与输出

KV cache 保存每层旋转后的 K 和未旋转的 V。模型相同、上下文追加一个 token 时复用；Sample 统一提交所有层。上下文改变或截断会重算，阶段定位保留兼容的已提交历史。

Attention 结果包含头映射、query 位置、计算阶段与逐行 mask 诊断，子地图读取这些快照。Concat 只在 W_o 前发生一次。

`view/head_controls.py` 提供统一 head 选择控件，`Session.select_head` 同步父层和子地图，不执行计算。W_o 快照包含各 head 的输出贡献与合并结果。`RoomSpec.operation_indices` 将精简后的解码路线映射到原图表操作；存在未来 key 需要屏蔽时仍保留 mask 检查点。

`view/output.py` 分开 logits 与概率。`output_summary` 选择 Top 7 并累计省略概率，不重新归一化。默认句子按自身顺序推进，`Session(guided=False)` 使用实际抽样。

Attention 观察状态位于 `simulation/attention_probe.py`，视图只读已存诊断。`RoomSpec.label_keys` 提供独立于算子标识的中英文节点名。详细说明在 `content/attention_guide.py`。TeX 公式写在 `content/attention_math.md` 的 `$$` 块中，`python -m tools.build_attention_assets` 使用 LaTeX 和 dvipng 编译。`view/formulas.py` 从 `assets/formulas` 读取打包的 PNG。

## 检查

命令见 [开发与贡献](../CONTRIBUTING.zh-CN.md)。测试覆盖数值、缓存一致性、场景切换、输入保护和两种语言；Canvas 预览与隐藏 Tk 检查不能代替实际试玩。
