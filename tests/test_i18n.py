import re
import tkinter as tk
import unittest
from dataclasses import replace
from string import Formatter

from tokenwalk.app import Game
from tokenwalk.i18n import Locale, TEXTS
from tokenwalk.model.prompt import demo_config, quote_tokens
from tokenwalk.model.forward import ForwardPass
from tokenwalk.simulation.maps import group_for, checkpoints
from tokenwalk.simulation.world import Session


class LocalizationTests(unittest.TestCase):
    def test_templates_have_matching_fields_in_both_languages(self):
        for key, pair in TEXTS.items():
            self.assertEqual(len(pair), 2, key)
            fields = [{field for _, field, _, _ in Formatter().parse(s) if field}
                      for s in pair]
            self.assertEqual(*fields, key)
        with self.assertRaises(ValueError):
            Locale('fr')
        with self.assertRaises(KeyError):
            Locale().text('missing')

    def test_english_quote_uses_words_and_only_reveals_first_token(self):
        config = demo_config(language='en')
        tokens = tuple(config.vocab[i] for i in config.context)
        self.assertEqual(tokens, ('Machines', 'take', 'me', 'by', 'surprise',
                                  'with', 'great', 'frequency', '.'))
        self.assertEqual(tokens, quote_tokens('en'))
        session = Session(config, guided=True, language='en')
        self.assertEqual(session.token, 'Machines')
        self.assertEqual(session.sentence_cells, ('Machines',) + (None,) * 8)
        self.assertEqual(session.prompt_text,
                         'Machines take me by surprise with great frequency.')


class BilingualInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.game = Game(self.root, animate=False)
        self.game.canvas.winfo_width = lambda: 960
        self.game.canvas.winfo_height = lambda: 660

    def tearDown(self):
        self.game.close()

    def test_held_language_key_resets_active_forward_once_and_r_keeps_locale(self):
        g = self.game
        g.fast_forward()
        g.session.update(.1)
        event = tk.Event()
        event.keysym = 'l'
        g.press(event)
        g.press(event)
        self.assertEqual(g.language, 'en')
        self.assertEqual(g.session.scene, 'macro')
        self.assertEqual(g.session.forward.next_index, 0)
        self.assertEqual(g.session.token, 'Machines')
        g.restart()
        self.assertEqual(g.session.language, 'en')
        self.assertEqual(g.session.token, 'Machines')
        g.release(event)
        g.press(event)
        self.assertEqual(g.session.token, '机')

    def test_language_switch_preserves_custom_model_and_layer_configuration(self):
        g = self.game
        custom = replace(g.config, blocks=('moe', 'dense', 'moe'))
        g.close()
        self.root = tk.Tk()
        self.root.withdraw()
        self.game = Game(self.root, custom, animate=False)
        self.game.toggle_language()
        self.assertEqual(self.game.config, custom)

    def test_language_switch_during_transfer_discards_pending_old_token(self):
        g = self.game
        old = g.session
        old.start_fast_forward()
        for _ in range(1000):
            old.update(1/60)
            if old.scene == 'birth':
                break
        self.assertEqual(old.scene, 'birth')
        self.assertIsNotNone(old.pending_token)
        g.toggle_language()
        g.session.update(.1)
        self.assertEqual(g.session.generation, 0)
        self.assertIsNone(g.session.pending_token)
        self.assertIsNone(g.session.transfer)
        self.assertEqual(g.session.sentence_cells, ('Machines',) + (None,) * 8)

    def test_every_english_room_and_message_contains_no_untranslated_chinese(self):
        g = self.game
        g.toggle_language()
        s = g.session
        # A full prefix exercises cropping, long token labels and both FFN types.
        s.forward = ForwardPass(s.initial_config)
        s.generation = len(s.initial_config.context) - 1
        for index, stage in enumerate(s.forward.stages):
            s.group = group_for(s.forward.stages, index)
            s.scene = 'micro'
            cp = next(c for c in checkpoints(s.group) if c.index == index)
            s.x, s.y = cp.x, cp.y
            s._execute_step(index)
            g.draw()
            texts = [g.canvas.itemcget(i, 'text') for i in g.canvas.find_all()
                     if g.canvas.type(i) == 'text']
            self.assertFalse(any(re.search(r'[\u4e00-\u9fff]', t) for t in texts), stage.kind)
        s.scene = 'done'
        g.draw()
        cells = g.canvas.find_withtag('sentence-cell')
        words = g.canvas.find_withtag('sentence-character')
        self.assertEqual(len(words), len(cells))
        for cell, word in zip(cells, words):
            cb, wb = g.canvas.bbox(cell), g.canvas.bbox(word)
            self.assertGreaterEqual(wb[0], cb[0])
            self.assertLessEqual(wb[2], cb[2])


if __name__ == '__main__':
    unittest.main()
