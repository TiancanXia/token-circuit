import unittest
from tokenwalk.simulation.world import Session
from tokenwalk.simulation.maps import checkpoints


def finish_detail(s):
    for cp in s.room_checkpoints:
        if cp.index >= s.rooms.active.progress:
            s.x, s.y = cp.x, cp.y
            s.interact()
    s.return_parent()


class DetailRoomTests(unittest.TestCase):
    def setUp(self):
        self.s = Session(guided=True, cache_enabled=True)
        self.s.jump_to(2)

    def test_third_level_commits_once_and_returns_to_same_module(self):
        s = self.s
        position = s.x, s.y
        self.assertTrue(s.interact())
        self.assertEqual(s.scene, 'detail')
        self.assertEqual(s.forward.next_index, 2)
        finish_detail(s)
        self.assertEqual(s.scene, 'micro')
        self.assertEqual((s.x, s.y), position)
        self.assertEqual(s.forward.next_index, 3)
        s.interact()
        self.assertTrue(s.rooms.active.complete)
        self.assertFalse(s.interact())
        self.assertEqual(s.forward.next_index, 3)

    def test_early_return_preserves_child_progress(self):
        s = self.s
        s.interact()
        s.interact()
        s.return_parent()
        s.interact()
        self.assertEqual(s.rooms.active.progress, 1)
        self.assertEqual(s.near_checkpoint.index, 1)
        self.assertEqual(s.forward.next_index, 2)
        finish_detail(s)
        self.assertEqual(s.forward.next_index, 3)

    def test_flight_and_jump_clear_child_state(self):
        s = self.s
        s.interact()
        s.interact()
        self.assertTrue(s.start_fast_forward())
        self.assertEqual(s.scene, 'flight')
        self.assertIsNone(s.rooms.active)
        s.jump_to(14)
        self.assertEqual(s.scene, 'micro')
        self.assertFalse(s.rooms.visits)
        self.assertFalse(s.rooms.parents)
        self.assertEqual(s.forward.next_index, 14)

    def test_all_children_and_layer_revisits(self):
        s = self.s
        for index in (2, 3, 4, 5):
            cp = next(c for c in checkpoints(s.group) if c.index == index)
            s.x, s.y = cp.x, cp.y
            s.interact()
            self.assertEqual(s.scene, 'detail')
            finish_detail(s)
            self.assertEqual(s.forward.next_index, index+1)
        self.assertEqual(len(s.rooms.visits), 4)

    def test_jump_preserves_compatible_history_for_cache_inspection(self):
        s=Session(guided=True,cache_enabled=True)
        s.start_fast_forward()
        for _ in range(1000):
            s.update(1/30)
            if s.generation==1:
                break
        self.assertEqual(s.forward.cache_reused,1)
        cells=s.sentence_cells
        s.jump_to(4)
        self.assertEqual(s.forward.cache_reused,1)
        self.assertEqual(s.forward.positions,[1])
        self.assertEqual(s.forward.next_index,4)
        self.assertEqual(s.sentence_cells,cells)
        s.interact()
        self.assertEqual(s.rooms.active.result.details['reused'],1)
        from tokenwalk.model.forward import ForwardPass
        full=ForwardPass(s.forward.config,s.seed+s.generation)
        for index in range(5):
            full.execute(index)
        for a,b in zip(s.rooms.active.result.values,full.results[4].values):
            self.assertAlmostEqual(a,b,places=13)
