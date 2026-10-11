"""Generate packaged math images; Matplotlib is only a developer dependency."""

from pathlib import Path
import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".local" / "formula-cache"),
)
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

FORMULAS = (
    r"s_j=\frac{q\cdot k_j}{\sqrt{d_h}}",
    r"p_j=\frac{\exp(s_j)}{\sum_{\ell\leq t}\exp(s_\ell)},\quad p_j=0\ (j>t)",
    r"o_h=\sum_{j\leq t}p_jv_j,\qquad O=\mathrm{Concat}(o_1,\ldots,o_H)W_o^T",
)


def main():
    output = Path(__file__).resolve().parents[1] / "assets" / "formulas"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams["mathtext.fontset"] = "cm"
    for index, formula in enumerate(FORMULAS):
        fig = plt.figure(figsize=(4.4, 0.68), dpi=200)
        fig.text(
            0.02,
            0.5,
            "$" + formula + "$",
            color="#a3cfdd",
            fontsize=17 if index == 0 else 13,
            va="center",
        )
        fig.savefig(output / f"attention-{index}.png", transparent=True)
        plt.close(fig)
    print("Three attention formula assets generated.")


if __name__ == "__main__":
    main()
