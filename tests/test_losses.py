import unittest

import torch

from losses import distillation_loss
from model import KeyboardDecoder, ModelConfig


class FakeTokenizer:
    def pad_id(self): return 0
    def unk_id(self): return 4
    def piece_to_id(self, piece): return {"<en>": 1, "<te>": 2, "<xlit>": 3}[piece]
    def encode(self, text, out_type=int):
        return [5 + (ord(char) % 40) for char in text.lower() if char.isalpha() or char == "'"]


class LossTests(unittest.TestCase):
    def test_distillation_loss_is_finite_and_differentiable(self):
        model = KeyboardDecoder(ModelConfig(vocab_size=48, d_model=24, layers=1,
                                             heads=4, ffn_dim=48, max_length=16, dropout=0))
        records = [{
            "mode": "en", "context": "how are", "target": "you",
            "candidates": ["you", "we", "they"],
            "teacher_log_probs": [-0.1, -2.0, -3.0],
        }]
        loss, parts = distillation_loss(model, records, FakeTokenizer(), 16, torch.device("cpu"),
                                        2.0, 0.65, 0.30, 0.05, True, 3)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.embedding.weight.grad)
        self.assertTrue(all(torch.isfinite(value) for value in parts.values()))


if __name__ == "__main__":
    unittest.main()
