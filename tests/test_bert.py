"""
Python file containing methods used for testing bert modules
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from settings import DATA_DIR
import bert.tokenization as bt


class TestTokenization:
    def test_full_tokenizer(self):
        vocab_path = os.path.join(DATA_DIR, "cased_L-12_H-768_A-12/vocab.txt")
        ft = bt.FullTokenizer(vocab_path)
        assert ft.tokenize("This is sample text to verify tokenizer.") == [
            "this",
            "is",
            "sample",
            "text",
            "to",
            "verify",
            "token",
            "##izer",
            ".",
        ]
