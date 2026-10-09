import unittest
from dataclasses import replace
from tokenwalk.model.architecture import ModelConfig
from tokenwalk.model.forward import ForwardPass
from tokenwalk.model.cache import KVCache
from tokenwalk.model.mathops import attention


def finish(f):
    while f.next_index < len(f.stages):
        f.execute(f.next_index)
    return f


class CacheAndGQATests(unittest.TestCase):
    def test_grouped_attention_matches_expanded_kv(self):
        q = [[1., 2., 3., 4., 5., 6., 7., 8.], [8., 7., 6., 5., 4., 3., 2., 1.]]
        k = [[.2, .4, .6, .8], [.8, .6, .4, .2]]
        v = [[1., 2., 3., 4.], [4., 3., 2., 1.]]
        expand = lambda rows: [r[:2] + r[:2] + r[2:] + r[2:] for r in rows]
        grouped, weights = attention(q, k, v, 4, 2)
        reference, original = attention(q, expand(k), expand(v), 4)
        self.assertEqual(grouped, reference)
        self.assertEqual(weights, original)
        self.assertTrue(all(head[0][1] == 0 for head in weights))

    def test_incremental_matches_full_prefix_for_mha_and_gqa(self):
        for kv in (4, 2, 1):
            cache = KVCache()
            config = ModelConfig(heads=4, kv_heads=kv, context=(0,))
            for token in (None, 1, 2, 0, 3):
                if token is not None:
                    config = replace(config, context=config.context + (token,))
                cached = finish(ForwardPass(config, cache=cache))
                full = finish(ForwardPass(config))
                for a, b in zip(cached.probabilities, full.probabilities):
                    self.assertAlmostEqual(a, b, places=13)
                self.assertEqual(cached.sampled, full.sampled)
                self.assertEqual(len(cached.x), 1)
                self.assertEqual(cached.cache_reused, len(config.context)-1)
                self.assertEqual(cache.context, config.context)
                self.assertEqual(len(cache.layers), len(config.blocks))

    def test_preview_and_partial_forward_do_not_publish_cache(self):
        cache = KVCache()
        f = ForwardPass(ModelConfig(context=(0,)), cache=cache)
        f.execute(0)
        before = f.current_vector
        f.preview(1)
        self.assertEqual(f.next_index, 1)
        self.assertEqual(f.current_vector, before)
        self.assertEqual(cache.context, ())
        self.assertFalse(cache.layers)

    def test_incompatible_or_non_append_context_recomputes(self):
        cache = KVCache()
        config = ModelConfig(heads=4, context=(0, 1))
        finish(ForwardPass(config, cache=cache))
        for changed in (replace(config, context=(1, 2)), replace(config, context=(0, 1, 2), kv_heads=2)):
            f = ForwardPass(changed, cache=cache)
            self.assertEqual(f.cache_reused, 0)
            finish(f)
            self.assertEqual(len(f.x), len(changed.context))

    def test_invalid_kv_heads(self):
        for kv in (0, 3, 8):
            with self.assertRaises(ValueError):
                ModelConfig(heads=4, kv_heads=kv)

    def test_rotated_history_unchanged_and_q_not_cached(self):
        cache=KVCache()
        config=ModelConfig(heads=4,kv_heads=2,context=(0,1,2))
        finish(ForwardPass(config,cache=cache))
        original={block:{key:[row[:] for row in rows] for key,rows in layer.items()} for block,layer in cache.layers.items()}
        f=ForwardPass(replace(config,context=(0,1,2,3)),cache=cache)
        finish(f)
        for block,layer in cache.layers.items():
            self.assertEqual(set(layer),{'k','v'})
            self.assertEqual(layer['k'][:-1],original[block]['k'])
            self.assertEqual(layer['v'][:-1],original[block]['v'])
        self.assertEqual(f.positions,[3])

    def test_window_rollover_invalidates_without_changing_probabilities(self):
        cache=KVCache()
        config=ModelConfig(heads=4,kv_heads=2,context=(0,1,2,3)*16)
        finish(ForwardPass(config,cache=cache))
        config=replace(config,context=config.context[1:]+(2,))
        f=finish(ForwardPass(config,cache=cache))
        reference=finish(ForwardPass(config))
        self.assertEqual(f.cache_reused,0)
        self.assertEqual(f.probabilities,reference.probabilities)
