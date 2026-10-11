import unittest
from dataclasses import replace
import tkinter as tk
from tokenwalk.app import Game
from tokenwalk.model.prompt import demo_config


class ReadabilityTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.g = Game(
            self.root,
            animate=False,
            guided=False,
            config=replace(demo_config(), context=(0, 1, 2, 3)),
        )
        self.g.canvas.winfo_width = lambda: 960
        self.g.canvas.winfo_height = lambda: 660
        self.g.jump_to(4)
        self.g.interact()

    def tearDown(self):
        self.g.close()

    def test_checkpoint_names_use_attention_terms_in_both_languages(self):
        expected = (
            "K / V",
            "Scores",
            "Causal mask",
            "Softmax",
            "Weighted sum",
            "Concat",
        )
        for language in ("zh", "en"):
            if self.g.language != language:
                self.g.toggle_language()
            self.g.jump_to(4)
            self.g.interact()
            s = self.g.session
            self.assertEqual(s.rooms.active.spec.nodes[-1], "Concat")
            self.assertEqual(tuple(s.checkpoint_title(i) for i in range(6)), expected)

    def test_inspector_labels_explain_content_contribution_result(self):
        s = self.g.session
        v = s.rooms.active
        v.progress = 3
        cp = v.spec.checkpoints()[3]
        s.x, s.y = cp.x, cp.y
        self.g.draw()
        words = [
            self.g.canvas.itemcget(i, "text")
            for i in self.g.canvas.find_all()
            if self.g.canvas.type(i) == "text"
        ]
        self.assertIn("Value V", words)
        self.assertIn("p × V", words)
        self.assertIn("Output o", words)
        self.assertNotIn("Σ pV", words)

    def test_mask_operation_keeps_its_name_when_no_mask_is_needed(self):
        g = self.g
        # Single-token full prefix has no future keys for its only query.
        g.config = replace(g.config, context=(0,))
        g.restart()
        g.jump_to(4)
        g.interact()
        self.assertEqual(g.session.checkpoint_title(2), "Causal mask")
        self.assertFalse(
            any(g.session.rooms.active.result.details["traces"][0]["mask_applied"])
        )
