import tkinter as tk
import unittest

from tokenwalk.app import Game
from tokenwalk.model.prompt import demo_config
from tokenwalk.view.output import output_summary


class OutputTests(unittest.TestCase):
    def test_default_vocabulary_sizes_are_16_and_12(self):
        self.assertEqual(len(demo_config(language='zh').vocab), 16)
        self.assertEqual(len(demo_config(language='en').vocab), 12)

    def test_top_display_retains_full_distribution_probabilities_and_remainder(self):
        probabilities = [i/136 for i in range(1, 17)]
        chosen, remainder = output_summary(probabilities)
        self.assertEqual(chosen, tuple(range(15, 8, -1)))
        self.assertAlmostEqual(sum(probabilities[i] for i in chosen) + remainder, 1.)
        self.assertGreater(remainder, 0.)
        self.assertEqual(output_summary([.25, .75]), ((1, 0), 0.))

    def test_logits_and_probabilities_are_labeled_at_their_own_nodes(self):
        root = tk.Tk()
        root.withdraw()
        g = Game(root, animate=False, language='en')
        g.canvas.winfo_width = lambda: 960
        g.canvas.winfo_height = lambda: 660
        try:
            indices = {stage.kind: i for i, stage in enumerate(g.session.forward.stages)
                       if stage.kind in ('unembedding', 'softmax')}
            for kind, index in indices.items():
                g.session.jump_to(index)
                g.interact()
                texts = [g.canvas.itemcget(i, 'text') for i in g.canvas.find_all()
                         if g.canvas.type(i) == 'text']
                self.assertIn('Vocabulary: 12 tokens', texts)
                if kind == 'unembedding':
                    self.assertIn('Next-token logits · vocab 12', texts)
                    self.assertTrue(any('not normalized' in t for t in texts))
                else:
                    self.assertIn('Next-token probabilities · vocab 12', texts)
                    self.assertTrue(any('Top 7/12' in t and 'rest' in t for t in texts))
                    self.assertAlmostEqual(sum(g.session.forward.probabilities), 1.)
        finally:
            g.close()


if __name__ == '__main__':
    unittest.main()
