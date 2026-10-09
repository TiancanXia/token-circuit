import unittest
from tokenwalk.model.mathops import rmsnorm, rope, attention, softmax, top_k


class OperatorTests(unittest.TestCase):
    def test_rms_does_not_center(self):
        out = rmsnorm([2., 2., 2., 2.])
        for v in out:
            self.assertAlmostEqual(v, 1., places=5)
        self.assertEqual(rmsnorm([0., 0.]), [0., 0.])

    def test_rope_preserves_norm_and_relative_dot_product(self):
        q, k = [1., 2., -3., 4.], [4., -1., 2., 3.]
        dot = lambda a, b: sum(x*y for x, y in zip(a, b))
        self.assertAlmostEqual(dot(rope(q, 3), rope(q, 3)), dot(q, q))
        self.assertAlmostEqual(dot(rope(q, 5), rope(k, 3)), dot(rope(q, 2), k))

    def test_causal_attention_ignores_future_values(self):
        q = k = [[1., 0.], [1., 0.], [1., 0.]]
        v = [[1., 2.], [3., 4.], [999., 999.]]
        result, weights = attention(q, k, v, heads=1)
        self.assertEqual(weights[0][0], [1., 0., 0.])
        self.assertEqual(result[0], [1., 2.])
        self.assertEqual(result[1], [2., 3.])
        self.assertEqual(weights[0][1][2], 0.)
        for head in weights:
            for row in head:
                self.assertAlmostEqual(sum(row), 1.)

    def test_routing_selects_and_normalizes_exactly_two_experts(self):
        chosen, weights = top_k([1., 3., 2.], 2)
        self.assertEqual(chosen, [1, 2])
        self.assertAlmostEqual(sum(weights), 1.)
        self.assertGreater(weights[0], weights[1])
        self.assertAlmostEqual(sum(softmax([1000., 1001.])), 1.)


if __name__ == '__main__':
    unittest.main()
