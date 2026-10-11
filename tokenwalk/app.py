"""Window/input entry point. Numeric and movement rules live in separate modules."""

import argparse
from dataclasses import replace
import time
import tkinter as tk
from .model.prompt import demo_config
from .simulation.world import Session
from .simulation.navigation import StageNavigator
from .view.renderer import Renderer, BG
from .view.context import focused_stage
from .view.guide_window import GuideWindow
from .view.window_icon import install_window_icon
from .i18n import Locale

ACTION_KEYS = frozenset(
    (
        "e",
        "return",
        "kp_enter",
        "f",
        "g",
        "m",
        "tab",
        "r",
        "l",
        "h",
        "c",
        "n",
        "f11",
        "escape",
    )
)


class Game:
    def __init__(
        self,
        root,
        config=None,
        animate=True,
        guided=True,
        language="zh",
        blocks=("dense", "moe"),
    ):
        self.root = root
        Locale(language)  # Validate before constructing the window/session.
        self.language = language
        self.default_prompt = config is None
        self.config = config or demo_config(blocks, language)
        self.guided = guided
        root.title(Locale(language).text("window"))
        root.geometry("1260x854")
        root.minsize(960, 660)
        root.configure(bg=BG)
        self.window_icon = install_window_icon(root)
        self.canvas = tk.Canvas(root, bg=BG, highlightthickness=0, takefocus=True)
        self.canvas.pack(fill="both", expand=True)
        self.renderer = Renderer(self.canvas)
        self.seed = 17
        self.cache_enabled = True
        self.session = Session(
            self.config,
            self.seed,
            guided=self.guided,
            language=self.language,
            cache_enabled=self.cache_enabled,
        )
        self.navigation = None
        self.guide = None
        self.keys = set()
        self.active = True
        self.fullscreen = False
        self.timer = None
        self.closed = False
        self.last = time.perf_counter()
        root.bind("<KeyPress>", self.press)
        root.bind("<KeyRelease>", self.release)
        root.bind("<FocusOut>", self.blur)
        root.bind("<FocusIn>", lambda e: self.focus())
        self.canvas.bind("<Button-1>", lambda e: self.canvas.focus_set())
        self.canvas.tag_bind("interact", "<Button-1>", lambda e: self.interact())
        self.canvas.tag_bind(
            "fast-forward", "<Button-1>", lambda e: self.fast_forward()
        )
        self.canvas.tag_bind("language", "<Button-1>", lambda e: self.toggle_language())
        self.canvas.tag_bind(
            "attention-mode", "<Button-1>", lambda e: self.toggle_attention()
        )
        self.canvas.tag_bind("cache-mode", "<Button-1>", lambda e: self.toggle_cache())
        self.canvas.tag_bind(
            "navigation-open", "<Button-1>", lambda e: self.toggle_navigation()
        )
        self.canvas.tag_bind("navigation-select", "<Button-1>", self.navigation_click)
        self.canvas.tag_bind("model-guide", "<Button-1>", lambda e: self.open_guide())
        self.canvas.tag_bind("attention-choice", "<Button-1>", self.attention_click)
        self.canvas.bind("<Configure>", lambda e: self.draw())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.canvas.focus_set()
        if animate:
            self.timer = root.after(16, self.frame)

    def press(self, event):
        if self.guide:
            return "break"
        key = event.keysym.lower()
        if key in self.keys:
            return "break" if key == "tab" or self.navigation else None
        self.keys.add(key)
        if self.navigation and key not in ("r", "l", "f11"):
            if self.active:
                moves = {
                    "left": (-1, 0),
                    "right": (1, 0),
                    "up": (0, -1),
                    "down": (0, 1),
                }
                if key in moves:
                    self.navigation.move(*moves[key])
                elif key in ("return", "kp_enter", "e"):
                    self.jump_to(self.navigation.selected)
                elif key in ("g", "escape"):
                    self.toggle_navigation()
                self.draw()
            return "break"
        if key in ("e", "return", "kp_enter") and self.active:
            self.interact()
        elif key == "f" and self.active:
            self.fast_forward()
        elif key == "g":
            self.toggle_navigation()
        elif key == "m":
            if self.active:
                self.session.return_parent()
            self.stop_motion()
            self.draw()
        elif key == "tab":
            if self.active:
                self.session.cycle_room_layer()
            self.stop_motion()
            self.draw()
            return "break"
        elif key == "r":
            self.restart()
        elif key == "l":
            self.toggle_language()
        elif key == "h":
            self.toggle_attention()
        elif key == "c":
            self.toggle_cache()
        elif key == "n" and self.active:
            self.session.cycle_head()
            self.draw()
        elif key == "f11":
            self.fullscreen = not self.fullscreen
            self.root.attributes("-fullscreen", self.fullscreen)
        elif key == "escape":
            if self.fullscreen:
                self.fullscreen = False
                self.root.attributes("-fullscreen", False)
            elif self.session.scene in ("micro", "detail"):
                self.session.return_parent()
                self.draw()
            else:
                self.close()

    def release(self, event):
        self.keys.discard(event.keysym.lower())

    def interact(self):
        if not self.active or self.navigation or self.guide:
            return
        if self.session.scene == "done":
            self.restart()
            return
        self.session.interact()
        self.stop_motion()
        self.draw()

    def stop_motion(self):
        """Clear movement while retaining held action keys across scene changes."""
        self.keys.intersection_update(ACTION_KEYS)

    def fast_forward(self):
        if not self.active or self.navigation or self.guide:
            return
        self.session.start_fast_forward()
        self.stop_motion()
        self.draw()

    def restart(self):
        if self.guide:
            return
        self.navigation = None
        self.stop_motion()
        self.seed += 1
        self.session = Session(
            self.config,
            self.seed,
            guided=self.guided,
            language=self.language,
            cache_enabled=self.cache_enabled,
        )
        self.focus()
        self.draw()

    def toggle_language(self):
        if not self.active or self.guide:
            return
        self.language = "en" if self.language == "zh" else "zh"
        if self.default_prompt:
            self.config = replace(
                demo_config(self.config.blocks, self.language),
                heads=self.config.heads,
                kv_heads=self.config.kv_heads,
            )
        self.root.title(Locale(self.language).text("window"))
        self.restart()

    def toggle_attention(self):
        if not self.active or self.navigation or self.guide:
            return
        heads = self.config.heads
        grouped = max((n for n in range(1, heads) if heads % n == 0), default=heads)
        kv = grouped if self.config.num_kv_heads == heads else heads
        self.config = replace(self.config, kv_heads=kv)
        self.restart()

    def toggle_cache(self):
        if not self.active or self.navigation or self.guide:
            return
        self.cache_enabled = not self.cache_enabled
        self.restart()

    def toggle_navigation(self):
        if not self.active or self.guide:
            return
        self.navigation = (
            None
            if self.navigation
            else StageNavigator(
                self.session.forward.stages, self.session.forward.next_index
            )
        )
        self.stop_motion()
        self.session.moving = False
        self.last = time.perf_counter()
        self.draw()

    def jump_to(self, index):
        if not self.active or self.guide:
            return
        self.session.jump_to(index)
        self.navigation = None
        self.stop_motion()
        self.last = time.perf_counter()
        self.draw()

    def navigation_click(self, event):
        if self.active and self.navigation and not self.guide:
            for tag in self.canvas.gettags("current"):
                if tag.startswith("stage:"):
                    self.jump_to(int(tag.split(":")[1]))
                    break

    def attention_click(self, event):
        for tag in self.canvas.gettags("current"):
            if tag.startswith("probe:"):
                _, action, value = tag.split(":")
                self.select_attention(action, int(value))
                break

    def select_attention(self, action, value):
        if not self.active or self.guide or self.navigation:
            return
        s = self.session
        if s.scene not in ("micro", "detail"):
            return
        if action in ("head", "select_head"):
            if action == "head":
                s.cycle_head(value)
            else:
                s.select_head(value)
            self.draw()
            return
        stage, result = focused_stage(s)
        if stage.kind != "attention" or result is None:
            return
        probe = s.attention_probe
        row, key = probe.indices(result.details)
        if action == "key" and 0 <= value < len(result.details["k_matrix"]):
            probe.key = value
        elif action == "row":
            probe.row = (row + value) % len(result.details["positions"])
        elif action == "page":
            probe.key = min(
                max(0, key + 6 * value), len(result.details["k_matrix"]) - 1
            )
        self.draw()

    def open_guide(self):
        """Open a read-only snapshot; only the document owns its scroll/focus."""
        if self.guide:
            self.guide.window.lift()
            return
        if not self.active or self.navigation:
            return
        stage, result = focused_stage(self.session)
        phase = result.details.get("phase") if result else None
        phase = phase or self.session.forward.attention_metadata()["phase"]
        topic = "overview" if self.session.scene == "macro" else stage.kind
        self.stop_motion()
        self.session.moving = False
        self.guide = GuideWindow(
            self.root,
            self.session.forward.config,
            self.language,
            self.guided,
            self.cache_enabled,
            phase,
            topic,
            self.renderer.cn,
            self.renderer.mono,
            self.close_guide,
        )

    def close_guide(self):
        self.guide = None
        self.keys.clear()
        if not self.closed:
            self.canvas.focus_set()
            self.focus()
            self.draw()

    def blur(self, event=None):
        self.active = False
        self.keys.clear()

    def focus(self):
        self.active = True
        self.last = time.perf_counter()

    def draw(self, dt=0.0):
        self.renderer.draw(
            self.session, not self.active, dt, navigation=self.navigation
        )

    def frame(self):
        self.timer = None
        if self.closed:
            return
        now = time.perf_counter()
        dt, self.last = now - self.last, now
        if self.active and self.navigation is None and self.guide is None:
            dx = int(bool({"right", "d"} & self.keys)) - int(
                bool({"left", "a"} & self.keys)
            )
            dy = int(bool({"down", "s"} & self.keys)) - int(
                bool({"up", "w"} & self.keys)
            )
            before = (self.session.scene, self.session.generation)
            self.session.update(dt, dx, dy)
            if before != (self.session.scene, self.session.generation):
                self.stop_motion()
        self.draw(dt)
        self.timer = self.root.after(25, self.frame)

    def close(self):
        self.closed = True
        if self.timer:
            self.root.after_cancel(self.timer)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Walk through a tiny decoder network.")
    parser.add_argument(
        "--blocks", default="dense,moe", help="Comma-separated dense/moe block types"
    )
    parser.add_argument(
        "--lang",
        choices=("zh", "en"),
        default="zh",
        help="Interface and preset quote language",
    )
    args = parser.parse_args()
    root = tk.Tk()
    Game(root, blocks=tuple(args.blocks.split(",")), language=args.lang)
    root.mainloop()
