# Code structure

[English](architecture.md) | [简体中文](architecture.zh-CN.md) · [Documentation](README.md)

## Files

| Location | Contents |
| --- | --- |
| `game.py`, `start.cmd` | Launch |
| `tokenwalk/app.py` | Tk window, input, focus and timers |
| `tokenwalk/i18n.py` | Chinese / English UI text |
| `tokenwalk/content/guide.py` | Module explanations |
| `tokenwalk/model/architecture.py` | Configuration and stages |
| `tokenwalk/model/mathops.py` | Numerical operators |
| `tokenwalk/model/forward.py` | Forward execution and result snapshots |
| `tokenwalk/model/cache.py` | Per-layer KV cache |
| `tokenwalk/model/prompt.py` | Sentence and vocabulary |
| `tokenwalk/simulation/` | Maps, rooms, movement, flight and transfer |
| `tokenwalk/view/` | Canvas drawing, sidebar and document window |
| `tools/`, `tests/` | Previews and verification |
| `assets/showcase/` | README images and animations |

Model code has no UI dependency. Simulation receives model results and a time step; views draw state. `app.py` handles window events.

## Execution

Manual E and fast-forward F use `Session._execute_step`. `ForwardPass.execute` checks stage order. Each configured block runs even though the world map displays one repeated-block frame.

After Sample, the character transfers from Output to Embedding. The sentence and context advance once after the 1.4-second animation. Processing the last sentence token ends the journey.

`Session.jump_to(index)` reconstructs the current prefix's earlier stages, cancels flight/transfer and opens the selected stage. It does not advance the sentence. G uses actual stage/block indices; N changes the inspected Q head, and Tab selects a reached layer.

Opening the handbook or losing window focus pauses progression. The handbook also blocks game actions until closed. Result selection in `view/context.py` is shared by the document and sidebar.

## Module rooms

`simulation/maps.py` contains the road graph and module groups. `simulation/rooms.py` registers `RoomSpec` checkpoints and stores visits and parent positions. `view/module_views.py` draws module rooms; `view/detail_views.py` draws registered children.

`ForwardPass.preview` reads a completed result or calculates a detached preview of the current stage. It does not change the executor, random state or cache. The final child checkpoint commits the stage once. M returns one level; F returns to the world route before continuing.

Stage diagnostics are stored in `Result.details`. Paired interface labels are in `i18n.py`; detailed topics and aliases are in `content/guide.py`.

## Cache and output

KV cache stores each layer's rotated K and unrotated V. It is reused when the model matches and the context appends one token. Sample commits all layers together. Changed or truncated contexts are recomputed. Stage jumps retain compatible committed history.

Attention results contain head mapping, query positions, phase and per-row mask diagnostics. Child drawings read these snapshots. Concat occurs once before W_o.

`view/output.py` separates logits from probabilities. `output_summary` selects Top 7 and sums omitted probability without renormalizing. The default sentence follows its own sequence; `Session(guided=False)` uses the actual sample.

Attention inspection state lives in `simulation/attention_probe.py`; its view reads stored traces. `RoomSpec.label_keys` provides translated checkpoint names independently of operation identifiers. Detailed content is in `content/attention_guide.py`. TeX formulas are authored as `$$` blocks in `content/attention_math.md`; `python -m tools.build_attention_assets` compiles them with LaTeX and dvipng. `view/formulas.py` loads the packaged PNGs from `assets/formulas`.

## Checks

See [contributing](../CONTRIBUTING.md) for commands. Tests cover numerical results, cache equivalence, scene transitions, input guards and both languages. Canvas previews and hidden Tk checks do not replace a visible playtest.
