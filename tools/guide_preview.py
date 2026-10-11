"""Optional reconstruction of the actual handbook widgets; not a screenshot."""

import tkinter as tk
from PIL import Image, ImageDraw
from tokenwalk.app import Game
from tools.preview import OUTPUT, preview_font


def capture_guide(game, name):
    guide = game.guide
    window = guide.window
    window.update()
    assert window.winfo_width() >= 620 and window.winfo_height() >= 460
    image = Image.new("RGB", (window.winfo_width(), window.winfo_height()), "#101e2c")
    draw = ImageDraw.Draw(image)

    def point(widget):
        return (
            widget.winfo_rootx() - window.winfo_rootx(),
            widget.winfo_rooty() - window.winfo_rooty(),
        )

    def widgets(parent):
        for widget in parent.winfo_children():
            yield widget
            yield from widgets(widget)

    for widget in widgets(window):
        if isinstance(widget, (tk.Label, tk.Button)):
            x, y = point(widget)
            assert 0 <= y and y + widget.winfo_height() <= window.winfo_height(), (
                widget,
                y,
            )
            face, size = window.tk.splitlist(widget.cget("font"))[:2]
            if isinstance(widget, tk.Button):
                draw.rectangle(
                    (x, y, x + widget.winfo_width(), y + widget.winfo_height()),
                    fill=widget.cget("bg"),
                )
            draw.text(
                (x + 3, y + 6),
                widget.cget("text"),
                fill=widget.cget("fg"),
                font=preview_font(face == game.renderer.mono, abs(int(size))),
            )
    x, y = point(guide.selector)
    draw.rectangle(
        (x, y, x + guide.selector.winfo_width(), y + guide.selector.winfo_height()),
        fill="#253e51",
    )
    draw.text(
        (x + 9, y + 2),
        guide.selector.get() + "   ▾",
        font=preview_font(False, 16),
        fill="#e4dfca",
    )
    text = guide.text
    x, y = point(text)
    draw.rectangle(
        (x, y, x + text.winfo_width(), y + text.winfo_height()), fill=text.cget("bg")
    )
    index = text.index("@0,0 display linestart")
    while text.compare(index, "<", "end-1c"):
        info = text.dlineinfo(index)
        following = text.index(index + " +1 display lines display linestart")
        if following == index:
            break
        if info:
            px, py, width, height, _ = info
            assert px + width <= text.winfo_width(), (index, info)
            style = next(
                (
                    tag
                    for tag in text.tag_names(index)
                    if tag in ("title", "heading", "formula")
                ),
                None,
            )
            font = text.tag_cget(style, "font") if style else text.cget("font")
            face, size = window.tk.splitlist(font)[:2]
            color = text.tag_cget(style, "foreground") if style else text.cget("fg")
            value = text.get(index, following).rstrip("\n")
            if 0 <= py and py + height <= text.winfo_height():
                draw.text(
                    (x + px, y + py),
                    value,
                    fill=color,
                    font=preview_font(face == game.renderer.mono, abs(int(size))),
                )
        for kind, image_name, at in text.dump(index, following, image=True):
            bounds = text.bbox(at)
            if kind == "image" and bounds and guide.formula_images:
                for number, formula in enumerate(guide.formula_images):
                    if str(formula) == image_name:
                        from tokenwalk.view.formulas import ASSETS

                        source = Image.open(ASSETS / f"attention-{number}.png").convert(
                            "RGBA"
                        )
                        source = source.resize(
                            (formula.width(), formula.height()),
                            Image.Resampling.LANCZOS,
                        )
                        bx, by, bw, bh = bounds
                        if 0 <= by < text.winfo_height():
                            source = source.crop(
                                (
                                    0,
                                    0,
                                    min(source.width, text.winfo_width() - bx),
                                    min(source.height, text.winfo_height() - by),
                                )
                            )
                            image.paste(source, (x + bx, y + by), source)
        index = following
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT / name)


def main():
    root = tk.Tk()
    root.withdraw()
    game = Game(root, animate=False)
    try:
        for language in ("zh", "en"):
            if game.language != language:
                game.toggle_language()
            game.jump_to(4)
            game.open_guide()
            # A transient of a withdrawn root stays unmapped. This developer
            # tool briefly maps the real document to inspect widget geometry.
            game.guide.window.transient("")
            game.guide.window.deiconify()
            capture_guide(game, f"preview-guide-{language}.png")
            game.guide.text.yview_moveto(1.0)
            capture_guide(game, f"preview-guide-bottom-{language}.png")
            game.guide.window.geometry("620x460")
            capture_guide(game, f"preview-guide-minimum-{language}.png")
            game.guide.close()
    finally:
        game.close()
    print("Handbook previews rendered in both languages at default and minimum size.")


if __name__ == "__main__":
    main()
