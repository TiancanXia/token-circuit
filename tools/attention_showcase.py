"""Render the attention instrument from a real guided six-token decode."""

import tkinter as tk
from tokenwalk.app import Game
from tools.preview import capture, verify_panel


def prepare(game):
    s = game.session
    for _ in range(5):
        s.start_fast_forward()
        generation = s.generation
        for _ in range(600):
            s.update(1 / 30)
            if s.generation > generation:
                break
        else:
            raise AssertionError("Forward did not finish")
    index = next(
        i for i, stage in enumerate(s.forward.stages) if stage.kind == "attention"
    )
    game.jump_to(index)
    game.interact()
    for cp in s.rooms.active.spec.checkpoints()[:3]:
        s.x, s.y = cp.x, cp.y
        game.interact()
    cp = s.rooms.active.spec.checkpoints()[3]
    s.x, s.y = cp.x, cp.y
    game.select_attention("key", 2)
    return s


def main():
    root = tk.Tk()
    root.withdraw()
    game = Game(root, animate=False)
    for language in ("zh", "en"):
        if game.language != language:
            game.toggle_language()
        prepare(game)
        for width, height in ((1260, 854), (960, 660)):
            game.canvas.winfo_width = lambda: width
            game.canvas.winfo_height = lambda: height
            game.draw()
            verify_panel(game)
            capture(game, f"attention-instrument-{language}-{width}.png")
        game.open_guide()
        game.guide.window.update_idletasks()
        assert len(game.guide.formula_images) == 3
        game.guide.close()
    game.close()
    print("Bilingual instruments and formula widgets verified at both window sizes.")


if __name__ == "__main__":
    main()
