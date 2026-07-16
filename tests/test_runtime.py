import unittest

import torch

from runtime import mps_available, resolve_device, resolve_precision


class RuntimeTests(unittest.TestCase):
    def test_explicit_cpu(self):
        device = resolve_device("cpu")
        self.assertEqual(device.type, "cpu")
        self.assertEqual(resolve_precision("auto", device), ("fp32", None))

    def test_cpu_rejects_fp16(self):
        with self.assertRaises(RuntimeError):
            resolve_precision("fp16", torch.device("cpu"))

    def test_mps_probe_returns_boolean(self):
        self.assertIsInstance(mps_available(), bool)


if __name__ == "__main__":
    unittest.main()
