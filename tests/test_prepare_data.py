import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PrepareDataTests(unittest.TestCase):
    def test_streamed_outputs_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            raw = base / "raw.txt"
            raw.write_text("\n".join(
                f"this is conversational message number {index}" for index in range(100)
            ) + "\n", encoding="utf-8")
            hashes = []
            for run in ("first", "second"):
                output = base / run
                command = [
                    sys.executable, str(ROOT / "prepare_data.py"),
                    "--source", "smoke", "--input", str(raw),
                    "--output-dir", str(output), "--keep-fraction", "0.4",
                ]
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                summary = json.loads(result.stdout)
                self.assertEqual(summary["scanned"], 100)
                self.assertGreater(sum(summary["counts"].values()), 0)
                hashes.append(tuple(
                    (output / f"smoke.{split}.jsonl").read_bytes()
                    for split in ("train", "dev", "test")
                ))
            self.assertEqual(hashes[0], hashes[1])


if __name__ == "__main__":
    unittest.main()
