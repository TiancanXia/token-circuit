"""Render the shared pixel character into a native Tk window icon."""

import tkinter as tk

from .glyphs import character
from .painting import BG


class _IconPainter:
    def __init__(self, image):
        self.image = image

    def rect(self, x, y, w, h, color):
        self.image.put(color, to=(x, y, x + w, y + h))

    def oval(self, x, y, w, h, color):
        cx, cy = x + w / 2, y + h / 2
        for py in range(y, y + h):
            for px in range(x, x + w):
                if ((px + 0.5 - cx) / (w / 2)) ** 2 + (
                    (py + 0.5 - cy) / (h / 2)
                ) ** 2 <= 1:
                    self.image.put(color, to=(px, py))


def install_window_icon(root):
    """Set the default icon for game windows; the caller retains the returned image."""
    pixels = tk.PhotoImage(master=root, width=32, height=32)
    pixels.put(BG, to=(0, 0, 32, 32))
    character(_IconPainter(pixels), 16, 25)
    icon = pixels.zoom(2)
    root.iconphoto(True, icon)
    return icon
