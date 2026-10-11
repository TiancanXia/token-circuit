import unittest
from dataclasses import replace

from tokenwalk.model.mathops import attention_trace
from tokenwalk.model.prompt import demo_config
from tokenwalk.simulation.maps import checkpoints
from tokenwalk.simulation.rooms import specification
from tokenwalk.simulation.world import Session


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

    def cached_decode(self):
        s = Session(guided=True, cache_enabled=True, language="en")
        s.start_fast_forward()
        for _ in range(1000):
            s.update(1 / 30)
            if s.generation == 1:
                break
        self.assertEqual(s.generation, 1)
        s.jump_to(4)
        self.assertTrue(s.interact())
        return s

    def test_maskless_cached_decode_has_five_sequential_checkpoints(self):
        s = self.cached_decode()
        visit = s.rooms.active
        self.assertEqual(visit.result.details["phase"], "decode")
        self.assertFalse(
            any(any(trace["mask_applied"]) for trace in visit.result.details["traces"])
        )
        self.assertEqual(
            tuple(s.checkpoint_title(cp.index) for cp in s.room_checkpoints),
            ("K / V", "Scores", "Softmax", "Weighted sum", "Concat"),
        )
        self.assertEqual(tuple(cp.index for cp in s.room_checkpoints), (0, 1, 2, 3, 4))
        self.assertEqual(visit.spec.operation_indices, (0, 1, 3, 4, 5))
        self.assertEqual(
            tuple(visit.spec.operation_index(cp.index) for cp in s.room_checkpoints),
            (0, 1, 3, 4, 5),
        )

    def test_maskless_cached_decode_completes_once_and_revisits_safely(self):
        s = self.cached_decode()
        visit = s.rooms.active
        self.assertEqual(len(s.room_checkpoints), 5)
        for cp in s.room_checkpoints[:-1]:
            s.x, s.y = cp.x, cp.y
            self.assertTrue(s.interact())
            self.assertEqual(s.forward.next_index, 4)
        cp = s.room_checkpoints[-1]
        s.x, s.y = cp.x, cp.y
        self.assertTrue(s.interact())
        self.assertTrue(visit.complete)
        self.assertEqual(s.forward.next_index, 5)
        self.assertEqual(len(s.forward.results), 5)
        self.assertIs(visit.result, s.forward.results[4])
        results = tuple(s.forward.results)
        self.assertFalse(s.interact())
        self.assertTrue(s.return_parent())
        self.assertTrue(s.interact())
        self.assertIs(s.rooms.active, visit)
        self.assertEqual(visit.progress, 5)
        self.assertTrue(visit.complete)
        self.assertFalse(s.interact())
        self.assertEqual(s.forward.next_index, 5)
        self.assertEqual(tuple(s.forward.results), results)

    def test_prefill_and_cache_off_keep_mask_checkpoint_even_for_single_query(self):
        for cache_enabled in (True, False):
            for context in ((0,), (0, 1, 2)):
                with self.subTest(cache_enabled=cache_enabled, context=context):
                    s = Session(
                        config=replace(demo_config(), context=context),
                        cache_enabled=cache_enabled,
                        language="en",
                    )
                    s.jump_to(4)
                    self.assertTrue(s.interact())
                    visit = s.rooms.active
                    self.assertEqual(
                        visit.result.details["phase"],
                        "prefill" if cache_enabled else "full",
                    )
                    self.assertEqual(
                        tuple(s.checkpoint_title(cp.index) for cp in s.room_checkpoints),
                        (
                            "K / V",
                            "Scores",
                            "Causal mask",
                            "Softmax",
                            "Weighted sum",
                            "Concat",
                        ),
                    )
                    self.assertEqual(
                        tuple(visit.spec.operation_index(i) for i in range(6)),
                        (0, 1, 2, 3, 4, 5),
                    )

    def test_chunked_decode_with_future_keys_keeps_mask_checkpoint(self):
        s = Session(config=replace(demo_config(), context=(0, 1, 2, 3)))
        s.jump_to(4)
        result = s.forward.preview(4)
        data = result.details
        queries = data["q_matrix"][-2:]
        values, traces = attention_trace(
            queries,
            data["k_matrix"],
            data["v_matrix"],
            s.forward.config.heads,
            s.forward.config.num_kv_heads,
            query_positions=[2, 3],
        )
        chunk = replace(
            result,
            values=values[-1],
            details={
                **data,
                "q_matrix": queries,
                "positions": [2, 3],
                "reused": 2,
                "phase": "decode",
                "traces": traces,
            },
        )
        for trace in traces:
            self.assertEqual(trace["mask_applied"], [True, False])
            self.assertEqual(trace["weights"][0][3], 0)
        spec = specification(chunk, s.forward.config)
        self.assertEqual(len(spec.checkpoints()), 6)
        self.assertEqual(spec.nodes[2], "Causal mask")
        self.assertEqual(
            tuple(spec.operation_index(i) for i in range(6)), (0, 1, 2, 3, 4, 5)
        )

    def test_new_child_and_revisit_use_current_selected_head(self):
        s = self.s
        s.attention_probe.head = 2
        self.assertTrue(s.interact())
        visit = s.rooms.active
        self.assertEqual(visit.head, 2)
        visit.head = 3
        s.return_parent()
        s.attention_probe.head = 1
        self.assertTrue(s.interact())
        self.assertIs(s.rooms.active, visit)
        self.assertEqual(visit.head, 1)
