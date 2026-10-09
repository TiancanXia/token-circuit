import math
import unittest
from dataclasses import replace
from tokenwalk.model.architecture import ModelConfig, build_stages
from tokenwalk.model.forward import ForwardPass


class ForwardTests(unittest.TestCase):
    def test_future_token_cannot_change_earlier_positions_across_dense_and_moe(self):
        config = ModelConfig(context=(0, 1, 2, 3))
        original = self.complete(config)
        perturbed = self.complete(replace(config, context=(0, 1, 6, 7)))
        self.assertEqual(original.x[:2], perturbed.x[:2])
        self.assertNotEqual(original.x[2:], perturbed.x[2:])

    def complete(self, config=None, seed=17):
        run = ForwardPass(config or ModelConfig(), seed)
        while run.next_index < len(run.stages):
            run.execute(run.next_index)
        return run

    def test_full_forward_produces_one_normalized_sample(self):
        run = self.complete()
        self.assertEqual(len(run.results), 27)
        self.assertAlmostEqual(sum(run.probabilities), 1.)
        self.assertIn(run.sampled, run.config.vocab)
        self.assertTrue(all(math.isfinite(v) for r in run.results for v in r.values))
        self.assertEqual(run.sampled, self.complete().sampled)

    def test_residual_uses_original_stream_and_rope_leaves_v_unchanged(self):
        run = self.complete()
        for result in run.results:
            if result.stage.kind.startswith('residual'):
                expected = [a+b for a, b in zip(result.details['skip'], result.details['branch'])]
                self.assertEqual(result.values, expected)
                source_kind = 'embedding' if result.stage.block == 0 and result.stage.kind == 'residual_attn' else 'residual_ffn' if result.stage.kind == 'residual_attn' else 'residual_attn'
                source_block = -1 if source_kind == 'embedding' else result.stage.block-1 if result.stage.kind == 'residual_attn' else result.stage.block
                original = next(r for r in run.results if r.stage.kind == source_kind and r.stage.block == source_block)
                self.assertEqual(result.details['skip'], original.values)
            if result.stage.kind == 'rope':
                qkv = next(r for r in run.results if r.stage.kind == 'qkv' and r.stage.block == result.stage.block)
                self.assertEqual(result.details['v'], qkv.details['v'])

    def test_moe_weights_and_expert_outputs_are_consistent(self):
        run = self.complete()
        routed = next(r for r in run.results if r.stage.kind == 'router')
        combined = next(r for r in run.results if r.stage.kind == 'combine')
        self.assertEqual(len(routed.details['selected']), 2)
        self.assertAlmostEqual(sum(routed.details['weights']), 1.)
        expected = [sum(weight*combined.details['experts'][idx][d] for idx, weight in zip(routed.details['selected'], routed.details['weights'])) for d in range(8)]
        self.assertEqual(combined.values, expected)

    def test_skipping_and_revisiting_are_rejected_without_changes(self):
        run = ForwardPass(ModelConfig())
        with self.assertRaises(ValueError):
            run.execute(1)
        run.execute(0)
        with self.assertRaises(ValueError):
            run.execute(0)
        self.assertEqual(run.next_index, 1)

    def test_different_layer_types_and_dimensions_are_supported(self):
        config = ModelConfig(dim=12, heads=3, hidden=16, experts=4, blocks=('moe', 'dense', 'dense'))
        run = self.complete(config)
        self.assertEqual(len(run.x), len(config.context))
        self.assertTrue(all(len(row)==config.dim for row in run.x))
        self.assertEqual(len(run.probabilities), len(config.vocab))
        self.assertEqual(len(build_stages(config)), 38)

    def test_full_qkv_matrix_diagnostics_are_actual_and_immutable(self):
        from tokenwalk.model.prompt import demo_config
        run=ForwardPass(demo_config())
        for i in range(3):
            run.execute(i)
        qkv=run.results[2]
        self.assertEqual(len(qkv.details['q_matrix']),len(run.config.context))
        self.assertEqual(qkv.details['q_matrix'][-1],qkv.values)
        original=[row[:] for row in qkv.details['q_matrix']]
        run.execute(3)
        rotated=run.results[3]
        self.assertEqual(qkv.details['q_matrix'],original)
        self.assertEqual(rotated.details['v_matrix'],qkv.details['v_matrix'])
        self.assertNotEqual(rotated.details['q_matrix'][-1],original[-1])


if __name__ == '__main__':
    unittest.main()
