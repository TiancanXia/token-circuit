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

    def test_path_names_are_plain_localized_actions_without_changing_spec(self):
        s = self.g.session
        self.assertEqual(s.rooms.active.spec.nodes[-1], "Concat")
        self.assertEqual(s.checkpoint_title(4), "汇聚内容")
        self.assertEqual(s.checkpoint_title(2), "屏蔽未来")
        self.assertNotIn("Σ", s.checkpoint_title(4))

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
        self.assertIn("内容", words)
        self.assertIn("贡献", words)
        self.assertIn("汇聚结果", words)
        self.assertNotIn("Σ pV", words)

    def test_no_future_key_caption_explains_reading_existing_positions(self):
        g = self.g
        # Single-token full prefix has no future keys for its only query.
        g.config = replace(g.config, context=(0,))
        g.restart()
        g.jump_to(4)
        g.interact()
        self.assertEqual(g.session.checkpoint_title(2), "可读位置")
