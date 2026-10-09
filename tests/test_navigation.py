import tkinter as tk
import unittest
from unittest.mock import patch

from tokenwalk.app import Game
from tokenwalk.model.forward import ForwardPass
from tokenwalk.model.prompt import demo_config
from tokenwalk.simulation.navigation import StageNavigator
from tokenwalk.simulation.world import Session


class NavigationTests(unittest.TestCase):
    def test_picker_covers_every_stage_and_distinguishes_repeated_blocks(self):
        s = Session(guided=True)
        picker = StageNavigator(s.forward.stages, 0)
        self.assertEqual(sorted(i for col in picker.columns for i in col),
                         list(range(len(s.forward.stages))))
        self.assertEqual(len(picker.columns), 3)
        picker.move(1, 0)
        self.assertEqual(s.forward.stages[picker.selected].block, 0)
        picker.move(1, 0)
        self.assertEqual(s.forward.stages[picker.selected].block, 1)
        picker.move(0, 100)
        self.assertIn(picker.selected, picker.columns[picker.column])

    def test_jump_to_each_stage_computes_only_prerequisites_with_same_numbers(self):
        s = Session(guided=True)
        reference = ForwardPass(s.forward.config, s.seed)
        for i in range(len(reference.stages)):
            reference.execute(i)
        for index in reversed(range(len(reference.stages))):
            self.assertTrue(s.jump_to(index))
            self.assertEqual(s.forward.results, reference.results[:index])
            self.assertEqual(s.forward.next_index, index)
            self.assertEqual(s.near_checkpoint.index, index)
            self.assertEqual(s.scene, 'micro')
            self.assertEqual(s.generation, 0)
            self.assertEqual(s.sentence_cells[1:], (None,) * 8)
            self.assertTrue(s.can_walk(s.x, s.y))

    def test_jump_during_transfer_discards_pending_token_and_keeps_current_prefix(self):
        s = Session(guided=True)
        s.start_fast_forward()
        for _ in range(1000):
            s.update(1/60)
            if s.scene == 'birth':
                break
        self.assertEqual(s.scene, 'birth')
        context = s.forward.config.context
        target = next(i for i, stage in enumerate(s.forward.stages)
                      if stage.block == 1 and stage.kind == 'router')
        s.jump_to(target)
        self.assertEqual(s.forward.config.context, context)
        self.assertIsNone(s.transfer)
        self.assertIsNone(s.flight)
        self.assertIsNone(s.pending_token)
        s.update(.1)
        self.assertEqual(s.generation, 0)
        self.assertEqual(s.forward.next_index, target)

    def test_invalid_target_does_not_change_a_running_flight(self):
        s = Session(guided=True)
        s.start_fast_forward()
        before = s.forward
        for target in (-1, len(s.forward.stages), 'qkv'):
            with self.assertRaises(ValueError):
                s.jump_to(target)
            self.assertIs(s.forward, before)
            self.assertEqual(s.scene, 'flight')

    def test_reposition_keeps_revealed_prefix_and_f_finishes_exactly_one_round(self):
        s = Session(guided=True)
        s.start_fast_forward()
        for _ in range(1000):
            s.update(1/60)
            if s.generation == 1:
                break
        context = s.forward.config.context
        cells = s.sentence_cells
        s.jump_to(13)  # Q/K/V inside block 2.
        self.assertEqual(s.forward.config.context, context)
        self.assertEqual(s.sentence_cells, cells)
        self.assertEqual(s.generation, 1)
        s.start_fast_forward()
        for _ in range(1000):
            s.update(1/60)
            if s.generation == 2:
                break
        self.assertEqual(s.generation, 2)
        self.assertEqual(s.forward.config.context, s.initial_config.context[:3])
        self.assertEqual(len(s.generated), 2)

    def test_picker_scrolls_to_more_configured_blocks_without_losing_stages(self):
        for blocks in (('dense',), ('moe', 'dense', 'moe', 'dense')):
            s = Session(demo_config(blocks), guided=True)
            picker = StageNavigator(s.forward.stages, 0)
            for _ in blocks:
                picker.move(1, 0)
            self.assertEqual(s.forward.stages[picker.selected].block, len(blocks) - 1)
            self.assertIn(picker.column, picker.visible_columns)


class NavigationInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.game = Game(self.root, animate=False)
        self.game.canvas.winfo_width = lambda: 960
        self.game.canvas.winfo_height = lambda: 660

    def tearDown(self):
        self.game.close()

    def key(self, name):
        event = tk.Event()
        event.keysym = name
        self.game.press(event)
        self.game.release(event)

    def test_keyboard_selects_second_block_and_waits_before_executing_target(self):
        g = self.game
        self.key('g')
        self.key('Right')
        self.key('Right')
        self.key('Down')
        target = g.navigation.selected
        self.assertEqual(g.session.forward.stages[target].block, 1)
        self.key('Return')
        self.assertIsNone(g.navigation)
        self.assertEqual(g.session.forward.next_index, target)
        self.assertEqual(g.session.near_checkpoint.index, target)
        self.key('e')
        self.assertEqual(g.session.scene, 'detail')
        self.assertEqual(g.session.forward.next_index, target)
        from test_detail_rooms import finish_detail
        finish_detail(g.session)
        self.assertEqual(g.session.forward.next_index, target + 1)

    def test_open_picker_pauses_flight_and_esc_closes_only_picker(self):
        g = self.game
        g.fast_forward()
        self.key('g')
        clock = g.session.flight.clock
        with patch('tokenwalk.app.time.perf_counter', return_value=g.last + .1):
            g.frame()
        self.assertEqual(g.session.flight.clock, clock)
        self.key('Escape')
        self.assertIsNone(g.navigation)
        self.assertFalse(g.closed)
        self.assertEqual(g.session.scene, 'flight')

    def test_picker_is_bilingual_bounded_and_closes_on_language_reset(self):
        g = self.game
        for language in ('zh', 'en'):
            if g.language != language:
                g.toggle_language()
            self.key('g')
            g.draw()
            rows = g.canvas.find_withtag('navigation-row')
            self.assertEqual(len(rows), len(g.session.forward.stages))
            for row in rows:
                box = g.canvas.bbox(row)
                self.assertTrue(0 <= box[0] < box[2] <= 960)
                self.assertTrue(0 <= box[1] < box[3] <= 660)
            g.toggle_language()
            self.assertIsNone(g.navigation)


if __name__ == '__main__':
    unittest.main()
