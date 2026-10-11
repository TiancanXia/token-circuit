import unittest
from pathlib import Path
from tools.build_attention_assets import extract_math, latex_document


class FormulaSourceTests(unittest.TestCase):
    def test_markdown_math_blocks_keep_tex_structure(self):
        formulas = extract_math("# Example\n\n$$\n\\frac{a}{b}\n$$\n")
        self.assertEqual(formulas, [r"\frac{a}{b}"])
        source = latex_document(formulas)
        self.assertIn(r"\usepackage{amsmath,amssymb}", source)
        self.assertIn(r"\frac{a}{b}", source)

    def test_packaged_source_has_score_probability_and_vector_sum(self):
        source = (
            Path(__file__).resolve().parents[1] / "tokenwalk/content/attention_math.md"
        )
        formulas = extract_math(source.read_text(encoding="utf-8"))
        self.assertEqual(len(formulas), 3)
        self.assertIn(r"\sqrt{d_h}", formulas[0])
        self.assertIn(r"\begin{cases}", formulas[1])
        self.assertIn(r"\sum", formulas[2])
