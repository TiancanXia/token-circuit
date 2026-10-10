# 开发与贡献

[English](CONTRIBUTING.md) | [简体中文](CONTRIBUTING.zh-CN.md) · [首页](README.zh-CN.md)

## 环境

```sh
git clone https://github.com/TiancanXia/token-circuit.git
cd token-circuit
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python game.py
```

需要 Python 3.10+ 与 Tkinter。Ruff 和 Pillow 用于开发，试玩无需安装。

## 同步与验证

修改前：

```sh
git status
git fetch origin
git pull --ff-only
```

拉取前保存本地改动，较大修改使用分支；历史分叉时保留双方工作并解决冲突。

修改后：

```sh
python -m unittest discover -s tests -v
python check_ui.py
python -m ruff check tokenwalk game.py check_ui.py preview.py --select F
python -m ruff format --check tokenwalk
python game.py
```

试玩受影响的操作并检查两种语言。模型修改还需核对缓存与完整前缀计算是否一致。

```sh
git diff --check
git diff
git add <changed-files>
git commit -m "Describe the change"
git push
```

外部贡献者请 fork、创建分支并提交 PR。

## 参考

- [代码结构](docs/architecture.zh-CN.md)
- [模型计算](docs/model-audit.zh-CN.md)

修改文档或界面文字时同步两种语言。英文使用默认文件名，中文使用 `.zh-CN.md`。

不提交凭据、缓存与临时文件。生成预览放在 `assets/previews`，README 展示资源保存在跟踪的 `assets/showcase`。

贡献内容需兼容 MIT；第三方内容保留来源和授权。
