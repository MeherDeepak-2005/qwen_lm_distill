import tempfile
import unittest
from pathlib import Path

from utils import read_jsonl, write_jsonl


class UtilsTests(unittest.TestCase):
    def test_atomic_jsonl_and_gzip_outputs(self):
        for filename in ("rows.jsonl", "rows.jsonl.gz"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / filename
                count = write_jsonl(output, iter(({"value": 1}, {"value": 2})), atomic=True)
                self.assertEqual(count, 2)
                self.assertEqual(list(read_jsonl(output)), [{"value": 1}, {"value": 2}])
                self.assertEqual(list(output.parent.glob("*.partial*")), [])
