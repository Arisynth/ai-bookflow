from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bookflow.memory import TranslationMemory, parse_term_response


class TranslationMemoryTests(unittest.TestCase):
    def test_persistent_terms_and_rolling_context(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "translation_memory.json"
            memory = TranslationMemory.load(path, book_title="Test", configured_terms={"Elizabeth": "伊丽莎白"})
            memory.add_terms({"Elizabeth": "伊莉莎白"})
            self.assertEqual(memory.terms["Elizabeth"], "伊丽莎白")
            memory.add_terms({"Lucerne": "卢塞恩"})
            memory.record("s1", "chapter-1", "伊丽莎白走进卢塞恩。")
            context = memory.context_for(chapter="chapter-1", source_text="Elizabeth visits Lucerne.")
            self.assertIn("Elizabeth => 伊丽莎白", context)
            self.assertIn("Lucerne => 卢塞恩", context)
            self.assertIn("伊丽莎白走进卢塞恩", context)
            memory.save()
            restored = TranslationMemory.load(path, book_title="Test")
            self.assertEqual(restored.terms["Elizabeth"], "伊丽莎白")
            self.assertEqual(restored.chapter_summaries, {})
            self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["fingerprint"])

    def test_term_response_accepts_json_object_or_list(self):
        self.assertEqual(parse_term_response('{"terms":[{"source":"Beaufort","target":"博福特"}]}'), {"Beaufort": "博福特"})
        self.assertEqual(parse_term_response('[{"source":"Lucerne","target":"卢塞恩"}]'), {"Lucerne": "卢塞恩"})
        self.assertEqual(parse_term_response("not json"), {})


if __name__ == "__main__":
    unittest.main()
