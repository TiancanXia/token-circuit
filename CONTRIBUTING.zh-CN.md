# 开发与贡献

[English](CONTRIBUTING.md) | [简体中文](CONTRIBUTING.zh-CN.md) · [首页](README.zh-CN.md)


### 本地开发

第一次克隆即可，后续继续使用同一个目录：

```sh
git clone https://github.com/TiancanXia/token-circuit.git
cd token-circuit
python game.py
```

运行游戏不需要 pip 依赖。可选开发工具包含 Ruff 和用于生成预览的 Pillow：

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

### 同步、修改、验证、提交

开始修改前，先检查本地状态，再查看远端更新：

```sh
git status
git fetch origin
git pull --ff-only
```

只在本地修改已妥善保存、当前分支可以安全快进时执行 pull。若历史分叉，先检查并合并或变基；不要用强制推送或重置覆盖他人工作。较大功能建议单独开分支。

修改后运行：

```sh
python -m unittest discover -s tests -v
python check_ui.py
python -m ruff check tokenwalk game.py check_ui.py preview.py --select F
python -m ruff format --check tokenwalk
python game.py
```

自动界面检查不代替实际试玩。至少检查移动、地图进出、一轮 F、H/C/L 切换和文档打开关闭。涉及模型时比较缓存与完整前缀计算，涉及文案时检查两种语言。

确认改动范围后提交并推送：

```sh
git diff --check
git diff
git add <修改过的文件>
git commit -m "Describe the change"
git push
```

仓库拥有者可维护 main；外部贡献者请 fork、创建分支并提交 PR。提交前再次同步远端，发生冲突时保留双方改动。没有后台自动同步或自动上传。

### 扩展与文档

先阅读 [架构说明](docs/architecture.zh-CN.md) 和 [模型核对](docs/model-audit.zh-CN.md)。模块与子模块由数据注册；不要在键盘处理里增加散落的模块判断。新增功能同时更新双语文案、玩家说明与相关验证。

公开文档在 `docs` 中正常跟踪。`docs/archive`、`docs/superpowers`、本地代理说明、缓存、虚拟环境与批量预览被忽略；精选展示资源在 `assets/showcase` 跟踪。不要提交凭据、个人路径或临时文件。

贡献需有权采用本项目的 MIT 授权。引用、图片、字体等第三方内容必须保留来源与授权。

英文为默认文档，中文版本使用 `.zh-CN.md` 后缀；同一文件不混写两份正文。修改时同步语言切换链接。MIT 许可证与更新日志只保留英文。
