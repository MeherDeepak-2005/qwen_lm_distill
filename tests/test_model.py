import unittest

import torch

from model import CompactInferenceModel, KeyboardDecoder, ModelConfig, count_parameters


class ModelTests(unittest.TestCase):
    def test_production_parameter_count(self):
        model = KeyboardDecoder(ModelConfig())
        self.assertEqual(count_parameters(model), 10_601_472)
        self.assertIs(model.head.weight, model.embedding.weight)

    def test_causal_prefix_is_unchanged_by_future_tokens(self):
        torch.manual_seed(2)
        model = KeyboardDecoder(ModelConfig(vocab_size=64, d_model=24, layers=2,
                                             heads=4, ffn_dim=48, max_length=8, dropout=0)).eval()
        left = torch.tensor([[1, 2, 3, 4, 5]])
        right = torch.tensor([[1, 2, 3, 9, 10]])
        with torch.inference_mode():
            a, b = model(left), model(right)
        torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=1e-6)

    def test_compact_wrapper_shapes(self):
        cfg = ModelConfig(vocab_size=64, d_model=24, layers=1, heads=4,
                          ffn_dim=48, max_length=8, dropout=0)
        wrapper = CompactInferenceModel(KeyboardDecoder(cfg).eval())
        batch = 3
        result = wrapper(
            torch.zeros((batch, 8), dtype=torch.int32),
            torch.zeros((batch, 8), dtype=torch.int32),
            torch.zeros((batch, 8), dtype=torch.int32),
            torch.ones((batch, 8)),
            torch.zeros((1, 1), dtype=torch.int32),
        )
        self.assertEqual(result[0].shape, (batch,))
        self.assertEqual(result[1].shape, (1, cfg.vocab_size))


if __name__ == "__main__":
    unittest.main()
