"""The model guide is a read-only, bilingual document over current configuration."""

import re
import tkinter as tk
import unittest
from unittest.mock import patch

from tokenwalk.app import Game
from tokenwalk.content.guide import TOPICS, document, topic_for
from tokenwalk.model.architecture import build_stages
from tokenwalk.model.prompt import demo_config


class GuideContentTests(unittest.TestCase):
    def test_every_stage_has_a_complete_bilingual_document(self):
        config = demo_config()
        for stage in build_stages(config):
            topic = topic_for(stage.kind)
            self.assertIn(topic, TOPICS)
            for language in ("zh", "en"):
                page = document(
                    topic,
                    config,
                    language,
                    guided=True,
                    cache_enabled=True,
                    phase="prefill",
                )
                self.assertGreaterEqual(len(page.sections), 4)
                self.assertTrue(all(title and body for title, body in page.sections))
                if language == "en":
                    self.assertFalse(re.search("[\u4e00-\u9fff]", str(page)))

    def test_document_distinguishes_cache_phases_and_probability_use(self):
        config = demo_config()
        decode = document("attention", config, "en", True, True, "decode")
        prefill = document("attention", config, "en", True, True, "prefill")
        full = document("attention", config, "en", True, False, "full")
        self.assertIn("Single-token decode", str(decode))
        self.assertIn("Prefill", str(prefill))
        self.assertIn("Full-prefix", str(full))
        self.assertIn("chunk", str(decode))
        self.assertIn(
            "journey text is independent",
            str(document("sample", config, "en", True, True, "decode")),
        )
        self.assertIn(
            "full vocabulary",
            str(document("sample", config, "en", False, False, "full")),
        )


class GuideInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.game = Game(self.root, animate=False)
        self.game.canvas.winfo_width = lambda: 960
        self.game.canvas.winfo_height = lambda: 660

    def tearDown(self):
        self.game.close()

    def test_button_opens_current_module_without_advancing_it(self):
        g = self.game
        g.jump_to(3)
        g.interact()
        before = (g.session.forward.next_index, g.session.rooms.active.progress)
        g.draw()
        self.assertTrue(g.canvas.find_withtag("model-guide"))
        g.open_guide()
        self.assertEqual(g.guide.topic, "rope")
        self.assertIn("RoPE", g.guide.text.get("1.0", "end"))
        self.assertEqual(
            before, (g.session.forward.next_index, g.session.rooms.active.progress)
        )
        first = g.guide
        g.open_guide()
        self.assertIs(g.guide, first)
        g.guide.close()
        self.assertIsNone(g.guide)
        self.assertEqual(
            before, (g.session.forward.next_index, g.session.rooms.active.progress)
        )

    def test_reading_pauses_flight_and_blocks_game_actions(self):
        g = self.game
        g.fast_forward()
        g.open_guide()
        g.focus()  # Even root focus cannot resume simulation behind the document.
        before = (g.session.flight.clock, g.seed, g.config, g.language)
        for action in (
            g.interact,
            g.fast_forward,
            g.toggle_attention,
            g.toggle_cache,
            g.toggle_language,
            g.toggle_navigation,
        ):
            action()
        event = tk.Event()
        event.keysym = "r"
        g.press(event)
        with patch("tokenwalk.app.time.perf_counter", return_value=g.last + 10):
            g.frame()
        self.assertEqual(before, (g.session.flight.clock, g.seed, g.config, g.language))
        g.guide.close()
        g.focus()
        with patch("tokenwalk.app.time.perf_counter", return_value=g.last + 0.1):
            g.frame()
        self.assertGreater(g.session.flight.clock, before[0])

    def test_document_topics_scroll_and_escape_closes_only_the_document(self):
        g = self.game
        g.toggle_language()
        g.open_guide()
        guide = g.guide
        for topic in TOPICS:
            guide.show_topic(topic)
            text = guide.text.get("1.0", "end")
            self.assertFalse(re.search("[\u4e00-\u9fff]", text))
            self.assertEqual(guide.text.cget("state"), "disabled")
        guide.window.update()
        guide.text.yview_moveto(1.0)
        self.assertGreater(guide.text.yview()[0], 0.0)
        guide.window.focus_force()
        guide.window.event_generate("<Escape>")
        guide.window.update()
        self.assertIsNone(g.guide)
        self.assertFalse(g.closed)
        self.assertEqual(g.session.forward.next_index, 0)


if __name__ == "__main__":
    unittest.main()
