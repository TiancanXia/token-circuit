import tkinter as tk
import unittest
from dataclasses import replace
from unittest.mock import patch
from tokenwalk.app import Game


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.game = Game(self.root, animate=False)
        self.game.canvas.winfo_width = lambda: 960
        self.game.canvas.winfo_height = lambda: 660

    def tearDown(self):
        self.game.close()

    def test_sentence_rail_hides_future_characters_and_long_explanations(self):
        g = self.game
        g.draw()
        c = g.canvas
        self.assertEqual(
            len(c.find_withtag("sentence-cell")), len(g.session.prompt_text)
        )
        glyphs = c.find_withtag("sentence-character")
        self.assertEqual(len(glyphs), 1)
        self.assertEqual(c.itemcget(glyphs[0], "text"), "机")
        texts = [c.itemcget(i, "text") for i in c.find_all() if c.type(i) == "text"]
        self.assertNotIn(g.session.forward.stages[0].explanation, texts)
        self.assertNotIn("输入：" + g.session.prompt_text, texts)

    def test_e_key_enters_immediately_and_hold_does_not_execute_again(self):
        e = tk.Event()
        e.keysym = "e"
        self.game.press(e)
        self.game.press(e)
        self.assertEqual(self.game.session.scene, "micro")
        self.assertEqual(self.game.session.forward.next_index, 0)
        self.game.release(e)
        self.game.press(e)
        self.assertEqual(self.game.session.forward.next_index, 1)

    def test_focus_loss_clears_motion_and_prevents_actions(self):
        e = tk.Event()
        e.keysym = "Right"
        self.game.press(e)
        self.game.blur()
        self.assertFalse(self.game.keys)
        e.keysym = "e"
        self.game.press(e)
        self.assertEqual(self.game.session.scene, "macro")

    def test_m_returns_from_room_without_a_second_toggle(self):
        self.game.session.interact()
        e = tk.Event()
        e.keysym = "m"
        self.game.press(e)
        self.game.press(e)
        self.assertEqual(self.game.session.scene, "macro")

    def test_revisited_unembedding_shows_its_logits_after_softmax(self):
        s = self.game.session
        while s.forward.next_index < len(s.forward.stages) - 1:
            s.forward.execute(s.forward.next_index)
        logits = next(
            r.details["logits"] for r in s.forward.results if "logits" in r.details
        )
        index = s.forward.next_index
        s.x, s.y = 500, 555
        s.interact()
        self.game.draw()
        c = self.game.canvas
        texts = [c.itemcget(i, "text") for i in c.find_all() if c.type(i) == "text"]
        self.assertIn(f"{max(logits):+.2f}", texts)
        self.assertEqual(s.forward.next_index, index)

    def test_waiting_stage_does_not_show_previous_stages_sidebar_bars(self):
        g = self.game
        g.canvas.winfo_width = lambda: 900
        g.canvas.winfo_height = lambda: 610
        g.jump_to(2)  # RMSNorm has run, but the selected QKV has not.
        self.assertTrue(g.session.forward.current_vector)
        c = g.canvas
        graph = [
            item
            for item in c.find_all()
            if c.type(item) == "rectangle"
            and c.coords(item)[0] >= 662
            and 347 <= c.coords(item)[1] <= 375
            and c.itemcget(item, "fill") in ("#d9bb78", "#89b3c3")
        ]
        self.assertEqual(graph, [])

    def test_moe_keeps_selected_experts_visible_when_more_than_four_exist(self):
        g = self.game
        g.config = replace(g.config, experts=6, top_k=2)
        g.restart()
        target = next(
            i
            for i, stage in enumerate(g.session.forward.stages)
            if stage.kind == "experts"
        )
        g.jump_to(target)
        router = g.session.forward.results[target - 1]
        texts = [
            g.canvas.itemcget(i, "text")
            for i in g.canvas.find_all()
            if g.canvas.type(i) == "text"
        ]
        for expert in router.details["selected"]:
            self.assertIn(f"EXPERT {expert + 1}", texts)

    def test_moe_cropping_labels_omitted_selected_weight(self):
        g = self.game
        g.config = replace(g.config, experts=6, top_k=6)
        g.restart()
        target = next(
            i
            for i, stage in enumerate(g.session.forward.stages)
            if stage.kind == "experts"
        )
        g.jump_to(target)
        router = g.session.forward.results[target - 1]
        rest = sum(router.details["weights"][4:])
        texts = [
            g.canvas.itemcget(i, "text")
            for i in g.canvas.find_all()
            if g.canvas.type(i) == "text"
        ]
        crop = next((text for text in texts if "省略权重" in text), "")
        self.assertIn(f"{rest:.1%}", crop)

    def test_completion_button_restarts_same_guided_sentence(self):
        g = self.game
        while g.session.forward.next_index < len(g.session.forward.stages):
            g.session.forward.execute(g.session.forward.next_index)
        g.session.scene = "done"
        g.session.generation = len(g.session.initial_config.context) - 1
        g.interact()
        self.assertEqual(g.session.scene, "macro")
        self.assertEqual(g.session.generation, 0)
        self.assertEqual(g.session.token, "机")
        self.assertTrue(g.session.guided)

    def test_held_f_does_not_start_a_second_forward_after_transfer(self):
        g = self.game
        e = tk.Event()
        e.keysym = "f"
        g.press(e)
        self.assertEqual(g.session.scene, "flight")
        for _ in range(1000):
            with patch("tokenwalk.app.time.perf_counter", return_value=g.last + 0.1):
                g.frame()
            if g.session.generation == 1:
                break
        self.assertEqual(g.session.scene, "macro")
        g.press(e)
        self.assertEqual(g.session.scene, "macro")
        g.release(e)
        g.press(e)
        self.assertEqual(g.session.scene, "flight")

    def test_held_tab_still_suppresses_default_focus_navigation(self):
        e = tk.Event()
        e.keysym = "Tab"
        self.assertEqual(self.game.press(e), "break")
        self.assertEqual(self.game.press(e), "break")

    def test_focus_loss_pauses_flight_and_transfer(self):
        g = self.game
        g.session.start_fast_forward()
        g.blur()
        with patch("tokenwalk.app.time.perf_counter", return_value=g.last + 10):
            g.frame()
        self.assertEqual(g.session.forward.next_index, 0)
        self.assertEqual(g.session.flight.clock, 0.0)
        g.focus()
        for _ in range(1000):
            g.session.update(1 / 60)
            if g.session.scene == "birth":
                break
        self.assertEqual(g.session.scene, "birth")
        g.blur()
        with patch("tokenwalk.app.time.perf_counter", return_value=g.last + 10):
            g.frame()
        self.assertEqual(g.session.transfer.elapsed, 0.0)

    def test_held_r_resets_once_until_key_release(self):
        g = self.game
        event = tk.Event()
        event.keysym = "r"
        seed = g.seed
        g.press(event)
        g.press(event)
        self.assertEqual(g.seed, seed + 1)
        g.release(event)
        g.press(event)
        self.assertEqual(g.seed, seed + 2)


if __name__ == "__main__":
    unittest.main()
