"""Head inspection changes diagrams without changing the forward pass."""

import copy
from dataclasses import replace
import tkinter as tk
import unittest

from tokenwalk.app import Game
from tokenwalk.model.prompt import demo_config


class HeadSelectionTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.g = Game(
            self.root,
            config=replace(demo_config(), context=(0, 1, 2, 3), kv_heads=2),
            guided=False,
            animate=False,
        )
        self.g.canvas.winfo_width = lambda: 960
        self.g.canvas.winfo_height = lambda: 660

    def tearDown(self):
        self.g.close()

    def texts(self):
        return [
            self.g.canvas.itemcget(i, "text")
            for i in self.g.canvas.find_all()
            if self.g.canvas.type(i) == "text"
        ]

    def snapshot(self):
        s = self.g.session
        return (
            s.forward.next_index,
            s.generation,
            s.sentence_cells,
            copy.deepcopy(s.forward.results),
            copy.deepcopy(s.cache.layers),
            s.forward.rng.getstate(),
            s.rooms.active.progress if s.rooms.active else None,
        )

    def test_all_attention_children_offer_direct_head_selection_without_execution(self):
        g = self.g
        for language in ("zh", "en"):
            g.language = language
            g.restart()
            for stage in (2, 3, 4, 5, 13, 14, 15, 16):
                g.jump_to(stage)
                g.interact()
                s = g.session
                visit = s.rooms.active
                for cp in visit.spec.checkpoints():
                    s.x, s.y = cp.x, cp.y
                    g.interact()
                    before = self.snapshot()
                    g.select_attention("select_head", 3)
                    self.assertEqual(visit.head, 3)
                    self.assertEqual(s.attention_probe.head, 3)
                    self.assertTrue(g.canvas.find_withtag("probe:select_head:0"))
                    self.assertTrue(g.canvas.find_withtag("probe:select_head:3"))
                    self.assertEqual(self.snapshot(), before)
                    g.select_attention("select_head", 0)

    def test_parent_and_child_keep_the_selected_head_and_keyboard_cycles_parent(self):
        g = self.g
        g.jump_to(2)
        g.interact()
        for cp in g.session.room_checkpoints:
            g.session.x, g.session.y = cp.x, cp.y
            g.interact()
        g.select_attention("select_head", 2)
        g.session.return_parent()
        g.draw()
        self.assertIn("Q3", self.texts())
        self.assertTrue(g.canvas.find_withtag("probe:select_head:3"))
        event = tk.Event()
        event.keysym = "n"
        g.press(event)
        g.release(event)
        self.assertEqual(g.session.attention_probe.head, 3)
        g.interact()
        self.assertEqual(g.session.rooms.active.head, 3)

    def test_gqa_shared_kv_but_distinct_queries_are_displayed(self):
        g = self.g
        g.jump_to(2)
        g.interact()
        visit = g.session.rooms.active
        visit.progress = len(visit.spec.nodes)
        cp = visit.spec.checkpoints()[1]
        g.session.x, g.session.y = cp.x, cp.y
        displays = []
        for head in (0, 1, 2):
            g.select_attention("select_head", head)
            displays.append(
                [
                    (g.canvas.coords(i), g.canvas.itemcget(i, "fill"))
                    for i in g.canvas.find_withtag("head-values")
                ]
            )
        self.assertTrue(displays[0])
        self.assertNotEqual(displays[0], displays[1])
        cp = visit.spec.checkpoints()[3]
        g.session.x, g.session.y = cp.x, cp.y
        kv_displays = []
        for head in (0, 1, 2):
            g.select_attention("select_head", head)
            kv_displays.append(
                [
                    (g.canvas.coords(i), g.canvas.itemcget(i, "fill"))
                    for i in g.canvas.find_withtag("head-values")
                ]
            )
        self.assertEqual(kv_displays[0], kv_displays[1])
        self.assertNotEqual(kv_displays[1], kv_displays[2])

    def test_projection_head_contributions_sum_to_the_full_output(self):
        g = self.g
        g.jump_to(5)
        g.interact()
        visit = g.session.rooms.active
        contributions = visit.result.details.get("head_contributions", [])
        self.assertEqual(len(contributions), 4)
        for channel, expected in enumerate(visit.result.values):
            self.assertAlmostEqual(sum(v[channel] for v in contributions), expected)
        visit.progress = len(visit.spec.nodes)
        cp = visit.spec.checkpoints()[2]
        g.session.x, g.session.y = cp.x, cp.y
        g.select_attention("select_head", 3)
        self.assertTrue(g.canvas.find_withtag("head-values"))
        self.assertTrue(g.canvas.find_withtag("attention-total-output"))

    def test_modal_and_focus_guards_block_head_changes(self):
        g = self.g
        g.jump_to(2)
        g.interact()
        for block in ("focus", "guide", "navigation"):
            if block == "focus":
                g.blur()
            elif block == "guide":
                g.open_guide()
            else:
                g.toggle_navigation()
            g.select_attention("select_head", 2)
            self.assertEqual(g.session.rooms.active.head, 0)
            if block == "focus":
                g.focus()
            elif block == "guide":
                g.guide.close()
            else:
                g.toggle_navigation()

    def test_custom_head_pages_keep_later_heads_accessible(self):
        g = self.g
        g.config = replace(g.config, dim=24, heads=12, kv_heads=3)
        g.restart()
        g.jump_to(2)
        g.interact()
        visit = g.session.rooms.active
        visit.progress = len(visit.spec.nodes)
        cp = visit.spec.checkpoints()[4]
        g.session.x, g.session.y = cp.x, cp.y
        g.select_attention("select_head", 11)
        self.assertIn("Q12", self.texts())
        self.assertTrue(g.canvas.find_withtag("probe:select_head:11"))
        g.select_attention("head", 1)
        self.assertEqual(visit.head, 0)
        self.assertIn("Q1", self.texts())

    def test_waiting_parent_can_select_head_before_entering_a_child(self):
        g = self.g
        g.jump_to(2)
        self.assertTrue(g.canvas.find_withtag("probe:select_head:3"))
        before = self.snapshot()
        g.select_attention("select_head", 3)
        self.assertEqual(self.snapshot(), before)
        g.interact()
        self.assertEqual(g.session.rooms.active.head, 3)


if __name__ == "__main__":
    unittest.main()
