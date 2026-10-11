"""Compile Markdown math blocks with LaTeX into packaged transparent PNGs."""

from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tokenwalk" / "content" / "attention_math.md"


def extract_math(markdown):
    return re.findall(r"(?m)^\$\$[ \t]*\n([\s\S]*?)\n\$\$[ \t]*$", markdown)


def latex_document(formulas):
    pages = "\n\\newpage\n".join(r"\[" + formula + r"\]" for formula in formulas)
    return (
        r"\documentclass[12pt]{article}"
        "\n"
        r"\usepackage{amsmath,amssymb}"
        "\n"
        r"\pagestyle{empty}"
        "\n"
        r"\setlength{\textwidth}{24cm}"
        "\n"
        r"\begin{document}"
        "\n" + pages + "\n" + r"\end{document}"
        "\n"
    )


def run(command, directory):
    result = subprocess.run(
        command, cwd=directory, capture_output=True, text=True, timeout=60
    )
    if result.returncode:
        raise RuntimeError((result.stdout + result.stderr)[-4000:])


def main():
    # Pillow and TeX are build tools; the game only loads the resulting PNGs.
    from PIL import Image

    latex, dvipng = shutil.which("latex"), shutil.which("dvipng")
    if not latex or not dvipng:
        raise SystemExit("Install LaTeX and dvipng, and add them to PATH.")
    formulas = extract_math(SOURCE.read_text(encoding="utf-8"))
    if len(formulas) != 3:
        raise ValueError("Attention requires exactly three Markdown math blocks.")
    build = ROOT / ".local" / "formula-build"
    output = ROOT / "assets" / "formulas"
    build.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    (build / "attention.tex").write_text(latex_document(formulas), encoding="utf-8")
    run(
        [
            latex,
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "attention.tex",
        ],
        build,
    )
    run(
        [
            dvipng,
            "-D",
            "240",
            "-T",
            "tight",
            "-bg",
            "Transparent",
            "-fg",
            "rgb 0.64 0.81 0.87",
            "-o",
            "raw-%d.png",
            "attention.dvi",
        ],
        build,
    )
    for index in range(len(formulas)):
        with Image.open(build / f"raw-{index + 1}.png") as source:
            image = source.convert("RGBA")
        if image.width > 848:
            height = round(image.height * 848 / image.width)
            image = image.resize((848, height), Image.Resampling.LANCZOS)
        asset = Image.new("RGBA", (880, image.height + 32))
        asset.paste(image, (16, 16))
        asset.save(output / f"attention-{index}.png")
    print("Compiled three TeX formulas from attention_math.md.")


if __name__ == "__main__":
    main()
