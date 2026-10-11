"""Load pre-rendered formula assets with no runtime typesetting dependency."""

from pathlib import Path
import tkinter as tk

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "formulas"


def attention_images(master):
    images = []
    for index in range(3):
        path = ASSETS / f"attention-{index}.png"
        if not path.exists():
            return []
        images.append(tk.PhotoImage(master=master, file=str(path)).subsample(2, 2))
    return images
