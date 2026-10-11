"""Render the attention instrument from a real guided six-token decode."""

import argparse
import sys
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visible", action="store_true", help="Also verify actual visible windows"
    )
    args = parser.parse_args()
    if args.visible and sys.platform == "win32":
        import ctypes

        # Tk coordinates and screenshot pixels must use the same DPI space.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    root = tk.Tk()
    if not args.visible:
        root.withdraw()
    game = Game(root, animate=False)
    for language in ("zh", "en"):
        if game.language != language:
            game.toggle_language()
        prepare(game)
        for width, height in ((1260, 854), (960, 660)):
            if args.visible:
                root.geometry(f"{width}x{height}+80+60")
                root.update()
            else:
                game.canvas.winfo_width = lambda: width
                game.canvas.winfo_height = lambda: height
            game.draw()
            verify_panel(game)
            capture(game, f"attention-instrument-{language}-{width}.png")
            if args.visible:
                from PIL import ImageGrab
                from tools.preview import OUTPUT

                def screenshot():
                    x, y = game.canvas.winfo_rootx(), game.canvas.winfo_rooty()
                    ImageGrab.grab(bbox=(x, y, x + width, y + height)).save(
                        OUTPUT / f"attention-window-{language}-{width}.png"
                    )
                    root.quit()

                root.after(300, screenshot)
                root.mainloop()
        game.open_guide()
        game.guide.window.update_idletasks()
        assert len(game.guide.formula_images) == 3
        game.guide.close()
    game.close()
    print("Bilingual instruments and formula widgets verified at both window sizes.")


if __name__ == "__main__":
    main()
