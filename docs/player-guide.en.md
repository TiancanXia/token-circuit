# Player guide

[中文](player-guide.md) | [English](player-guide.en.md) · [Home](../README.md)

## Your first journey

1. Launch the game. Only the first token in the bottom sentence is lit.
2. Move toward Embedding with arrows or WASD; press E / Enter to enter.
3. Approach a device and press E to process it. Leave through the illuminated exit when finished.
4. Inside Attention, enter QKV, RoPE, Head attention and Output rooms for another level of detail. M returns one level and keeps your progress.
5. After the output stage, your character dissolves and reassembles at the beginning. The next token lights up. Repeat until the sentence ends.

F completes the current forward pass quickly. G opens the stage picker: left/right selects a block or input/output section, up/down selects a module, and Enter jumps there. A jump does not reveal the next token early.

## Controls

| Key | Action |
| --- | --- |
| Arrows / WASD | Move |
| E / Enter | Enter, process or leave |
| M / Esc | Return one level; Esc leaves fullscreen first, then exits on the world map |
| F | Complete the current pass quickly |
| G | Stage picker; G or Esc closes it |
| H | Switch MHA / GQA and restart |
| C | Toggle KV cache and restart |
| N / Tab | Inspect another Q head / reached layer |
| L / R / F11 | Change language / restart / fullscreen |

Click an explanation button to read a module document. Reading pauses the game; Esc closes only the document. Changing language restarts the journey and keeps the attention and cache settings.

## Questions

**Why are there black cells?** Future tokens are hidden. Each full pass reveals one more.

**Why isn't the next token the most probable one?** The journey follows a fixed sentence, independently of the probability chart. The chart contains actual results from an untrained miniature model. Top 7 preserves full-vocabulary probabilities without renormalizing them.

**The game will not start.** Python 3.10+, Tkinter and a graphical desktop are required. Try `python -m tkinter`. On Windows, the launcher keeps startup errors visible so you can include them in an issue.

**Do I need an API key or a model download?** No. Runtime uses only the Python standard library, offline.

**How can I report a problem?** Include your OS, Python version, reproduction steps and any errors in an [issue](https://github.com/TiancanXia/token-circuit/issues). Screenshots help.
