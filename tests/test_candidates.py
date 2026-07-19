import unittest

from candidates import BigramCandidates, WORD, WordTrie, merge_candidates


class CandidateTests(unittest.TestCase):
    def test_trie_typo_recall_and_frequency_order(self):
        trie = WordTrie()
        trie.insert("hello", 10)
        trie.insert("help", 3)
        trie.insert("yellow", 1)
        self.assertEqual(trie.search("helo", max_distance=1, limit=2), ["hello", "help"])

    def test_ngram(self):
        ngrams = BigramCandidates()
        ngrams.observe("how are you how are things how are you")
        self.assertEqual(ngrams.next_words("tell me how are", 2), ["you", "things"])

    def test_source_metadata_and_forced_gold(self):
        words, sources = merge_candidates(
            "we'll", {"trie": ["well", "we'll"], "qwen": ["go"]}, 4,
        )
        self.assertEqual(words[0], "we'll")
        self.assertEqual(sources["we'll"], ["gold", "trie"])

    def test_telugu_and_tenglish_words(self):
        self.assertEqual(WORD.findall("నేను ఇంటికి వస్తాను"), ["నేను", "ఇంటికి", "వస్తాను"])
        self.assertEqual(WORD.findall("nenu intiki vastanu"), ["nenu", "intiki", "vastanu"])
        trie = WordTrie()
        trie.insert("తెలుగు", 5)
        self.assertEqual(trie.search("తెలుగ", max_distance=1, limit=1), ["తెలుగు"])


if __name__ == "__main__":
    unittest.main()
