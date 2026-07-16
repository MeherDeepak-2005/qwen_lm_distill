import tempfile
import unittest
from pathlib import Path

from convert_source import dailydialog_jsonl_rows, gutenberg_rows, nus_xml_rows


class ConvertSourceTests(unittest.TestCase):
    def test_dailydialog_jsonl_preserves_dialogue_group(self):
        content = ('{"dialogue":[{"text":"hello"},{"text":"hi"}]}\n'
                   '{"dialogue":[{"text":"bye"}]}\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.json"
            path.write_text(content)
            rows = list(dailydialog_jsonl_rows(path))
        self.assertEqual([row["conversation_id"] for row in rows],
                         ["train:0", "train:0", "train:1"])

    def test_gutenberg_blank_lines_define_dialogues(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dialog.txt"
            path.write_text("hello\nhow are you\n\nfine thanks\n")
            rows = list(gutenberg_rows(path))
        self.assertEqual([row["conversation_id"] for row in rows], ["0", "0", "1"])
        self.assertEqual([row["turn"] for row in rows], [0, 1, 0])

    def test_nus_messages_group_by_sender(self):
        xml = """<smsCorpus>
          <message id="1"><text>Hello</text><source><userProfile><userID>51</userID></userProfile></source></message>
          <message id="2"><text>Again</text><source><userProfile><userID>51</userID></userProfile></source></message>
          <message id="3"><text>Other</text><source><userProfile><userID>72</userID></userProfile></source></message>
        </smsCorpus>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sms.xml"
            path.write_text(xml)
            rows = list(nus_xml_rows(path))
        self.assertEqual([row["conversation_id"] for row in rows], ["51", "51", "72"])


if __name__ == "__main__":
    unittest.main()
