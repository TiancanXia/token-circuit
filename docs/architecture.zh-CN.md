# 代码架构与扩展指南

[English](architecture.md) | [简体中文](architecture.zh-CN.md) · [文档目录](README.zh-CN.md)

## 包与职责

```text
game.py / start.cmd            稳定启动入口
tokenwalk/
  app.py                      Tk 生命周期、窗口焦点、输入锁与计时
  i18n.py                     成对的中英文界面模板与结果标签
  content/guide.py             只读双语模型文档与配置快照
  model/
    architecture.py           配置、计算阶段与描述
    mathops.py                纯数值算子
    forward.py                有序执行与不可变诊断快照
    cache.py                  已提交的逐层 KV 与兼容性判断
    prompt.py                 双语句子、演示分词与词表
  simulation/
    maps.py                   装置入口、道路、模块分组与可行走区域
    world.py                  大地图、模块、子模块、飞行、转场与结束状态
    rooms.py                  子模块注册、访问进度与父层栈
    flight.py                 路线与时间，输出待执行阶段编号
    transfer.py               确定性的消散、漂移与聚拢位置
    navigation.py             纯阶段选择与光标逻辑
  view/
    painting.py / glyphs.py    Canvas 适配器与共享像素小人
    map_view.py               装置与人物
    room_view.py              两层内部地图共享的地板与检查点
    module_views.py           模块图形分发与非 Attention 图形
    charts.py                 复用的数值条与矩阵
    attention.py              Attention 概览与缓存下的矩形 Q/K
    attention_glyphs.py        头分组、拼接向量与真实遮罩
    detail_views.py            从计算诊断绘制注册的子模块
    navigation.py             双语阶段选择面板
    output.py                 logits、概率与 Top-N 摘要
    effects.py                轨迹、淡出与粒子
    renderer.py               场景组合与视口变换
    hud.py                    简短侧栏与句子格子
    context.py                共享的当前阶段选择
    guide_window.py           可滚动的只读 Tk 模型文档
tools/                        开发检查与预览工具
tests/                        数值、模拟和 Tk 回归验证
assets/previews/              自动生成的 Canvas 重建图，忽略提交
assets/showcase/              README 使用的精选图片与动图，正常跟踪
```

模型不导入界面或模拟层。模拟层消费模型结果，输出时间与位置，不导入 Tk，使用调用方传入的 dt。视图只消费状态，不执行计算步骤。`app.py` 管理焦点、按键、计时和重启。

文档内容单独放在 `content/guide.py`，通过配置快照生成，不运行算子。`guide_window.py` 负责目录、带标签正文和滚动控件，不导入 Session 或执行阶段。侧栏和文档通过 `view/context.py` 选择同一阶段；`hud.py` 独立管理侧栏与句子布局。尚未计算的阶段显示等待，不借用前一阶段向量。

侧栏的 `model-guide` 按钮打开文档：大地图对应架构概览，内部地图对应当前操作。Game 同时只拥有一个文档窗口；窗口存在时持续暂停模拟，不受焦点事件影响。主窗口不能在其后重启、切换配置、定位或处理节点。关闭时清空按键、重置计时并恢复 Canvas 焦点；打开期间 Esc 只关闭文档。阅读不推进句子、缓存提交或子模块进度。先预留底部空间，再扩展正文，保证最小窗口仍能看到关闭说明。

场景使用较大的标签与成对的 `brief.*` 短文案。完整概念、公式、维度与边界放在 Topic 数据；norm/residual 变体通过别名共用说明。新增阶段必须补齐双语 Topic 或别名，无需增加窗口或输入分支。数值图保留必要的位置与裁剪标记；大型 MoE 的四个可见槽优先显示选中专家，并标注省略数量和路由权重。

## 执行与状态

手动 E 与自动 F 共用 `Session._execute_step`；`ForwardPass.execute` 拒绝跳过或重复阶段。飞行从当前 `next_index` 创建路线，沿地板和绘制共用的道路图行进，每个剩余编号只输出一次。重复 block 只压缩地图，实际计算逐层执行。内部地图中的 F 先回到大地图位置，再继续飞行。

Sample 时，引导模式选择下一预定 token，自由模式使用实际抽样。Output → Embedding 粒子转场持续 1.4 秒；转场完成后上下文与句子只追加一次，再回到大地图。最后一个预定 token 处理后进入 done，不再生成或转场。

动作键的按住状态跨自动场景切换保留，直到 KeyRelease；移动键清空。失去焦点暂停计算和动画并清空按键。R 按当前配置创建新 Session。特效只依赖模拟时间，缩放与重绘不推进动画。

G 打开 `StageNavigator`，按真实 `Stage.block` 分组全局阶段和各 block。打开时暂停模拟并接管移动、E 和 Esc；重启或语言切换会关闭它。视图最多显示三列，较长网络跟随选中 block 滚动。Canvas 行保存真实阶段编号，鼠标与键盘都调用 `Game.jump_to`。

`Session.jump_to(index)` 先验证，再以当前上下文和轮次种子创建新 ForwardPass，只计算目标之前的阶段。它取消飞行和转场，进入目标房间/检查点，并保存对应大地图入口；生成轮次与已亮格子不变。重建避免向后跳转遗留残差或诊断。之后 E/F 继续使用同一执行器。

`view/output.py` 分开绘制 Unembedding logits 与 Output 的全词表 Softmax 概率。`output_summary` 只选择展示项并累计省略概率，不重新归一化或修改分布。显示 Top 7/实际词表大小及剩余概率；输出侧栏计数为词表 token，而非特征通道。默认中文词表 16、英文 12，自定义模型动态读取大小。

## 语言与 token 标签

`Locale.text(key, **values)` 读取成对模板，缺少键或无效语言会报错。Session 翻译状态信息，Painter 共享同一 Locale；模型计算不依赖界面语言，技术名称与公式共用。数值诊断仅在视图翻译。

`prompt.demo_config(blocks, language)` 构造句子、词表与上下文：中文按字，英文按词与标点，属于演示规则，不是生产 tokenizer。`join_tokens` 恢复文字间距。句子格子不提前显示未来 token；`Painter.fit_text` 缩放单行文字并缩短窄矩阵标签，英文格子更宽。

L 或语言按钮重启整个 Session，避免混用不兼容词表或待生成 token。block 配置保留；显式传入的自定义 ModelConfig 原样保留，仅切换界面。R 保留语言。新增文案写入成对目录，不复制场景或数值处理器。

## 扩展方式

- 在模型配置与执行层添加算子，并编写纯数值验证。
- 在 `simulation/maps.py` 映射模块分组、入口和道路；地板、绘制与飞行共用该图。
- 在 `view/module_views.py` 注册模块图形，必要时拆出独立视图文件；装置外观放在 `map_view.py`。读取对应房间/层的快照，不取之后的全局结果。
- 添加内部地图时，在 `simulation/rooms.py` 注册 RoomSpec，在 `i18n.py` 配对文案，在 `view/detail_views.py` 注册图形。复用输入、行走与地板；新数值诊断放在 `Result.details`。
- 在 `i18n.py` 添加简短场景文案，在 `content/guide.py` 添加详细双语说明或共用别名。文档测试要求每个阶段都有完整双语内容。
- 时间特效的几何与状态放在 simulation，绘制放在 view；输入仍归 `app.py`。
- 自动图片与可选依赖放在开发工具中，保留稳定的根目录启动和验证入口。

现有房间布局面向当前微型 decoder，不是任意网络通用布局。扩展其他拓扑时应明确替换地图定义。运行时代码使用 Ruff 标准格式；当前说明保留在 docs，历史里程碑留在忽略的 docs/archive。预览与可选工具不能进入运行时导入链。

## 嵌套内部地图与缓存

RoomSpec 通过数据定义检查点标题与位置，Rooms 保存逐阶段访问记录及父层位置。到达父检查点后 E 使用 `ForwardPass.preview` 打开子模块：已完成阶段读取快照，当前阶段使用脱离真实状态的预计算。预览不改变执行器、随机数状态或缓存。局部检查点只推进 `Visit.progress`，最后一个检查点才调用共用的执行方法。重进保留进度，完成后可查看而不重放。M/Esc 返回一级；F 退出内部地图再沿大地图飞行；G 清除旧访问并重建前置计算。Tab 先回模块再切换已到达的层，N 只改变诊断中观察的 Q head。全屏下 Esc 会先退出全屏。

ModelConfig 区分 Q heads 与 KV heads：`head_dim=dim/heads`，`kv_dim=num_kv_heads*head_dim`；`attention_mode/kv_head_for` 将布局提供给共享图形。纯 `attention_trace` 一次计算分组 scores、可见性、概率与加权 V，普通 attention 包装同一实现。末尾有效 key 的 query 无需显式 mask，之前的 query 将未来 score 设为 −∞；逐行 `mask_applied` 驱动视图，也覆盖分块查询。矩阵按输出×输入存储，因此行向量公式是 XWᵀ。

MHA/GQA 配置整个 Attention，不是额外顺序算子。QKV 的 Heads 显示映射；Head attention 在 RoPE/KV 准备之后，结束于唯一的 Concat；Output · W_o 查看已拼接输入并投影。每份 Result 保存 prefill/decode/full 阶段，包括禁止发布缓存的脱离状态预览。标签读取真实 mask 诊断，不能只看缓存开关。等待中的父视图显示静态架构说明，不借用别的阶段图形。

KVCache 只保存完整提交的逐层旋转 K、未旋转 V、上下文元组和结构签名。只有模型不变、上下文恰好追加一个 token 时，ForwardPass 才复用历史；增量位置保留绝对前缀下标。每层暂存新合并 K/V，Sample 完成后统一发布。变更、非追加或截断上下文会重算。Game 默认开缓存，独立 Session 默认全前缀，也可传入 `cache_enabled=True`。定位保留兼容的已提交历史，因此可继续查看实际缓存；不兼容时重算，完整 forward 结束后才提交新缓存。

子模块图形只读快照，不执行阶段。两级内部地板共用 `room_checkpoints/checkpoint_complete/checkpoint_title`。当前明确支持三层地图和已注册模块内部地图，不是通用递归游戏引擎。

## 验证

测试覆盖完整前缀/增量概率、GQA/显式扩展 K/V、因果 mask、隔离预览、缓存兼容性、子模块进度与重访、飞行/定位取消、按键锁、双语图形和裁剪后的加权贡献。`tools.attention_preview` 还通过实际六 token 缓存解码重建两种语言，并对照关闭缓存的结果；Pillow 只作为可选开发依赖。

Canvas 重建图和隐藏 Tk 事件检查补充实际窗口验证，不能据此宣称人工键盘试玩完成。

```sh
python -m unittest discover -s tests -v
python check_ui.py
python -m ruff check tokenwalk game.py check_ui.py preview.py --select F
python -m ruff format --check tokenwalk
```

README 展示资源保存在跟踪的 assets/showcase；批量预览和历史计划保留本地并忽略。同步与发布流程见 [开发与贡献](../CONTRIBUTING.zh-CN.md)。
