"""Compose main map, module world, numerical sidebar and AR birth scene."""

import random
from tkinter import font
from .painting import Painter, BG, TEXT, MUTED, GOLD
from ..simulation.maps import houses
from .map_view import macro
from .room_view import draw_room
from .module_views import draw_module
from .hud import sidebar, sentence_rail
from .detail_views import draw_detail
from .effects import transfer
from .navigation import navigation_panel

WIDTH, HEIGHT = 900, 610
VIEW = (24, 142, 610, 357)


class Renderer:
    def __init__(self, canvas):
        self.canvas = canvas
        families = set(font.families(canvas))
        self.cn = next(
            (
                f
                for f in (
                    "Microsoft YaHei UI",
                    "Microsoft YaHei",
                    "PingFang SC",
                    "Noto Sans CJK SC",
                )
                if f in families
            ),
            "TkDefaultFont",
        )
        self.mono = next(
            (f for f in ("Consolas", "Courier New") if f in families), "TkFixedFont"
        )
        rng = random.Random(21)
        self.stars = [
            (rng.randrange(27, 631), rng.randrange(146, 494), rng.choice((1, 1, 2)))
            for _ in range(59)
        ]

    def world_painter(self, p, scene):
        vx, vy, vw, vh = VIEW
        left, top, right, bottom = (
            (0, 0, 760, 410) if scene in ("micro", "detail") else (20, 91, 779, 602)
        )
        zoom = min(vw / (right - left), vh / (bottom - top))
        offset = (
            p.ox + (vx + (vw - (right - left) * zoom) / 2 - left * zoom) * p.scale,
            p.oy + (vy + (vh - (bottom - top) * zoom) / 2 - top * zoom) * p.scale,
        )
        return Painter(p.c, p.cn, p.mono, p.scale * zoom, offset, p.locale.language)

    def draw(self, s, paused=False, dt=0.0, navigation=None):
        c = self.canvas
        c.delete("all")
        cw, ch = max(1, c.winfo_width()), max(1, c.winfo_height())
        scale = min(cw / WIDTH, ch / HEIGHT)
        p = Painter(
            c,
            self.cn,
            self.mono,
            scale,
            ((cw - WIDTH * scale) / 2, (ch - HEIGHT * scale) / 2),
            s.language,
        )
        for x, y, size in self.stars:
            p.rect(x, y, size, 1, "#355160")
        wp = self.world_painter(p, s.scene)
        if s.scene in ("micro", "detail"):
            draw_room(wp, s)
        else:
            macro(wp, s)
        if s.scene == "birth":
            transfer(wp, s)
        # Crop world drawing without making every glyph know about the viewport.
        p.rect(-1000, -1000, 3000, 1142, BG)
        p.rect(-1000, 499, 3000, 2000, BG)
        p.rect(-1000, 142, 1024, 357, BG)
        p.rect(634, 142, 2000, 357, BG)
        if s.scene == "micro":
            draw_module(p, s)
        elif s.scene == "detail":
            draw_detail(p, s)
        p.text(25, 16, p.tr("title"), 26, TEXT)
        if s.scene == "detail":
            house = next(h.title for h in houses() if h.key == s.group.house)
            level = p.tr(
                "detail_level",
                house=house,
                n=s.group.block + 1,
                title=s.forward.stages[s.rooms.active.index].title,
            )
        elif s.scene == "micro":
            title = next(h.title for h in houses() if h.key == s.group.house)
            level = p.tr("inside", title=title) + (
                " · " + p.tr("layer", n=s.group.block + 1) if s.group.block >= 0 else ""
            )
        else:
            level = p.tr(
                "complete"
                if s.scene == "done"
                else "macro"
                if s.scene == "macro"
                else "flight"
                if s.scene == "flight"
                else "birth"
            )
        p.fit_text(26, 53, level, 390, 12, GOLD)
        progress = (
            p.tr("progress", n=s.generation + 1, total=len(s.initial_config.context))
            if s.guided
            else p.tr("round", n=s.generation + 1)
        )
        p.text(876, 26, progress, 10, GOLD, anchor="ne")
        config = s.forward.config
        mode = config.attention_mode
        p.rect(436, 53, 212, 27, "#253e51", tags="attention-mode")
        p.fit_text(
            447,
            60,
            p.tr(
                "attention_mode", mode=f"{mode} {config.heads}Q/{config.num_kv_heads}KV"
            ),
            190,
            11,
            GOLD,
            tags="attention-mode",
        )
        p.rect(436, 88, 212, 26, "#1c3342", tags="cache-mode")
        p.fit_text(
            447,
            95,
            p.tr("cache_mode", state=p.tr("on" if s.cache_enabled else "off")),
            190,
            11,
            MUTED,
            tags="cache-mode",
        )
        p.rect(662, 53, 100, 27, "#253e51", tags="navigation-open")
        p.text(674, 60, p.tr("navigation_button"), 10, GOLD, tags="navigation-open")
        p.rect(776, 53, 100, 27, "#253e51", tags="language")
        p.text(790, 60, p.tr("language"), 10, GOLD, tags="language")
        p.fit_text(876, 98, p.tr("language_hint"), 214, 10, MUTED, anchor="ne")
        p.line([(24, 127), (876, 127)], "#314651")
        sidebar(p, s)
        message = p.tr("paused") if paused else s.message
        p.text(26, 513, message, 11, GOLD, width=606)
        sentence_rail(p, s)
        p.text(25, 594, p.tr("keys"), 8, MUTED)
        p.text(876, 594, p.tr("window_keys"), 7, MUTED, anchor="ne")
        if navigation:
            navigation_panel(p, s, navigation)
