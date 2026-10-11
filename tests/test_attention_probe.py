import unittest
from dataclasses import replace
from tokenwalk.model.prompt import demo_config
from tokenwalk.model.forward import ForwardPass
from tokenwalk.simulation.attention_probe import AttentionProbe, inspect_key


class ProbeTests(unittest.TestCase):
    def data(self, kv=2):
        config = replace(demo_config(), context=(0, 1, 2, 3), kv_heads=kv)
        forward = ForwardPass(config)
        index = next(i for i, s in enumerate(forward.stages) if s.kind == "attention")
        for i in range(index + 1):
            forward.execute(i)
        return config, forward.results[-1].details

    def test_contributions_reconstruct_every_grouped_head(self):
        config, data = self.data()
        for head in range(config.heads):
            parts = [inspect_key(data, config, head, -1, key) for key in range(4)]
            self.assertEqual(parts[0].kv_head, config.kv_head_for(head))
            for channel in range(config.head_dim):
                self.assertAlmostEqual(
                    sum(p.contribution[channel] for p in parts),
                    data["traces"][head]["outputs"][-1][channel],
                )

    def test_future_key_has_raw_score_but_zero_probability_and_contribution(self):
        config, data = self.data()
        item = inspect_key(data, config, 0, 0, 3)
        self.assertTrue(item.masked)
        self.assertEqual(item.probability, 0)
        self.assertTrue(all(v == 0 for v in item.contribution))

    def test_selection_clamps_to_new_prefix_and_is_read_only(self):
        probe = AttentionProbe(row=30, key=30)
        config, data = self.data()
        before = repr(data)
        self.assertEqual(probe.indices(data), (3, 3))
        inspect_key(data, config, 0, *probe.indices(data))
        self.assertEqual(repr(data), before)


class ProbeInterfaceTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from tokenwalk.app import Game

        self.root = tk.Tk()
        self.root.withdraw()
        self.game = Game(
            self.root,
            config=replace(demo_config(), context=(0, 1, 2, 3)),
            guided=False,
            animate=False,
        )
        self.game.canvas.winfo_width = lambda: 960
        self.game.canvas.winfo_height = lambda: 660
        self.game.jump_to(4)
        self.game.interact()
        visit = self.game.session.rooms.active
        visit.progress = 3
        cp = visit.spec.checkpoints()[3]
        self.game.session.x, self.game.session.y = cp.x, cp.y
        self.game.draw()

    def tearDown(self):
        self.game.close()

    def test_selection_updates_inspector_without_executing_or_advancing(self):
        g = self.game
        s = g.session
        before = (
            s.forward.next_index,
            s.rooms.active.progress,
            s.generation,
            repr(s.rooms.active.result.details),
        )
        g.select_attention("key", 2)
        self.assertEqual(s.attention_probe.key, 2)
        self.assertTrue(g.canvas.find_withtag("attention-choice"))
        g.select_attention("row", -1)
        self.assertEqual(s.attention_probe.row, 2)
        self.assertEqual(
            before,
            (
                s.forward.next_index,
                s.rooms.active.progress,
                s.generation,
                repr(s.rooms.active.result.details),
            ),
        )

    def test_document_and_navigation_guard_inspection(self):
        g = self.game
        g.open_guide()
        g.select_attention("key", 2)
        self.assertEqual(g.session.attention_probe.key, 0)
        self.assertTrue(g.guide.formula_images)
        g.guide.close()
        g.toggle_navigation()
        g.select_attention("key", 2)
        self.assertEqual(g.session.attention_probe.key, 0)

    def test_long_prefix_page_controls_do_not_cover_head_indicator(self):
        g = self.game
        for language in ("zh", "en"):
            g.language = language
            g.config = replace(
                demo_config(language=language), context=(0, 1, 2, 3, 4, 5, 6, 7)
            )
            g.restart()
            g.jump_to(4)
            g.interact()
            visit = g.session.rooms.active
            visit.progress = 3
            cp = visit.spec.checkpoints()[3]
            g.session.x, g.session.y = cp.x, cp.y
            for width, height in ((1260, 854), (960, 660)):
                g.canvas.winfo_width = lambda: width
                g.canvas.winfo_height = lambda: height
                g.draw()
                head = next(
                    i
                    for i in g.canvas.find_all()
                    if g.canvas.type(i) == "text"
                    and g.canvas.itemcget(i, "text").startswith("N / Q")
                )
                a = g.canvas.bbox(head)
                for i in g.canvas.find_withtag("attention-choice"):
                    if any(
                        tag.startswith("probe:page:") for tag in g.canvas.gettags(i)
                    ):
                        b = g.canvas.bbox(i)
                        self.assertTrue(
                            a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
                        )
            g.select_attention("page", 1)
            self.assertEqual(g.session.attention_probe.key, 6)
            g.select_attention("row", 1)
            g.select_attention("key", 7)
            data = visit.result.details
            item = inspect_key(
                data,
                g.session.forward.config,
                visit.head,
                *g.session.attention_probe.indices(data),
            )
            self.assertTrue(item.masked)
            self.assertEqual(item.probability, 0)
