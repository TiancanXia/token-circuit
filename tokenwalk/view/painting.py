"""Small Canvas adapter shared by map drawings, internal diagrams and HUD."""

from ..i18n import Locale

BG, TEXT, MUTED, GOLD = "#101e2c", "#e4dfca", "#9aafb5", "#d9bb78"


def mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return "#" + "".join(
        f"{round(int(a[i : i + 2], 16) * (1 - t) + int(b[i : i + 2], 16) * t):02x}"
        for i in (1, 3, 5)
    )


class Painter:
    def __init__(self, canvas, cn, mono, scale, offset=(0, 0), language="zh"):
        self.c, self.cn, self.mono, self.scale = canvas, cn, mono, scale
        self.ox, self.oy = offset
        self.locale = Locale(language)

    def tr(self, key, **values):
        return self.locale.text(key, **values)

    def point(self, x, y):
        return round(self.ox + x * self.scale), round(self.oy + y * self.scale)

    def rect(self, x, y, w, h, color, **kw):
        self.c.create_rectangle(
            *self.point(x, y), *self.point(x + w, y + h), fill=color, outline="", **kw
        )

    def oval(self, x, y, w, h, color, **kw):
        self.c.create_oval(
            *self.point(x, y), *self.point(x + w, y + h), fill=color, outline="", **kw
        )

    def poly(self, points, color):
        self.c.create_polygon(
            *[v for point in points for v in self.point(*point)], fill=color, outline=""
        )

    def line(self, points, color, width=1, dash=None):
        options = dict(fill=color, width=max(1, round(width * self.scale)))
        if dash:
            options["dash"] = tuple(max(1, round(v * self.scale)) for v in dash)
        self.c.create_line(
            *[v for point in points for v in self.point(*point)], **options
        )

    def text(
        self, x, y, value, size=8, color=TEXT, anchor="nw", mono=False, width=None, **kw
    ):
        options = dict(
            text=value,
            fill=color,
            anchor=anchor,
            font=(self.mono if mono else self.cn, -max(9, round(size * self.scale))),
        )
        if width:
            options["width"] = max(1, width * self.scale)
        self.c.create_text(*self.point(x, y), **options, **kw)

    def fit_text(
        self, x, y, value, width, size=8, color=TEXT, anchor="nw", mono=False, **kw
    ):
        """Fit one line; abbreviate tiny diagram labels, never wrap into cells."""
        face = self.mono if mono else self.cn
        pixels = max(9, round(size * self.scale))
        available = max(
            1, width * self.scale - 4
        )  # Canvas adds a small text bbox margin.
        value = str(value)

        def measured():
            return int(self.c.tk.call("font", "measure", (face, -pixels), value))

        while pixels > 9 and measured() > available:
            pixels -= 1
        if measured() > available:
            original = value
            for length in range(len(original) - 1, -1, -1):
                value = original[:length] + "…"
                if measured() <= available:
                    break
        self.c.create_text(
            *self.point(x, y),
            text=value,
            fill=color,
            anchor=anchor,
            font=(face, -pixels),
            **kw,
        )
