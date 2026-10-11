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

    def test_inspector_keeps_original_weights_when_selecting_a_token(self):
        s = self.g.session
        v = s.rooms.active
        v.progress = 3
        cp = v.spec.checkpoints()[3]
        s.x, s.y = cp.x, cp.y
        self.g.select_attention("key", 2)
        words = [
            self.g.canvas.itemcget(i, "text")
            for i in self.g.canvas.find_all()
            if self.g.canvas.type(i) == "text"
        ]
        weights = v.result.details["traces"][0]["weights"][-1]
        for weight in weights:
            self.assertIn(f"{weight:.1%}", words)

    def test_single_query_does_not_offer_query_switching(self):
        g = self.g
        g.config = replace(g.config, context=(0,))
        g.restart()
        g.jump_to(4)
        g.interact()
        visit = g.session.rooms.active
        visit.progress = 3
        cp = visit.spec.checkpoints()[3]
        g.session.x, g.session.y = cp.x, cp.y
        g.draw()
        tags = [tag for i in g.canvas.find_all() for tag in g.canvas.gettags(i)]
        self.assertFalse(any(tag.startswith("probe:row:") for tag in tags))
        self.assertTrue(any(tag.startswith("probe:head:") for tag in tags))

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

    def test_repeated_tokens_have_distinguishable_query_labels(self):
        g = self.g
        g.config = replace(g.config, context=(0, 0))
        g.restart()
        g.jump_to(4)
        g.interact()
        visit = g.session.rooms.active
        visit.progress = 3
        cp = visit.spec.checkpoints()[3]
        g.session.x, g.session.y = cp.x, cp.y
        labels = []
        for _ in range(2):
            g.select_attention("row", 1)
            labels.append(
                next(
                    g.canvas.itemcget(i, "text")
                    for i in g.canvas.find_all()
                    if g.canvas.type(i) == "text"
                    and g.canvas.itemcget(i, "text").startswith("Query")
                )
            )
        self.assertNotEqual(labels[0], labels[1])
