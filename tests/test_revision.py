from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bookflow.adapters import OllamaClient, OllamaReviewer, OllamaTranslator
from bookflow.config import Config
from bookflow.epub import EpubBook, Segment
from bookflow.pipeline import JobState, Pipeline


def verdict(ok=False, severity="minor"):
    return {"ok": ok, "severity": "none" if ok else severity,
            "issues": [] if ok else ["残留 bad"], "suggestion": "" if ok else "把 bad 改为正确中文"}


class RevisionTests(unittest.TestCase):
    def test_truncated_generation_is_retried_not_accepted(self):
        payloads = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return json.dumps({"done_reason": "length" if len(payloads) == 1 else "stop",
                                   "message": {"content": "partial" if len(payloads) == 1 else "complete"}}).encode()

        def opener(request, **kwargs):
            payloads.append(json.loads(request.data))
            return Response()

        client = OllamaClient("http://localhost:11434", "configured", opener=opener,
                              options={"num_predict": 1024}, max_retries=1, retry_backoff_seconds=0)
        self.assertEqual(client.chat("system", "user"), "complete")
        self.assertEqual(payloads[1]["options"]["num_predict"], 2048)
        self.assertEqual(client.options["num_predict"], 1024)

    def test_unfounded_rejection_is_verified_before_revision(self):
        calls = []

        class Client:
            def chat(self, system, user, **kwargs):
                calls.append((system, user))
                return json.dumps(verdict(len(calls) == 2))

        result = OllamaReviewer(Client()).review("source", "meaning already expressed", "context")
        self.assertTrue(result["ok"])
        self.assertFalse(result["preliminary_review"]["ok"])
        self.assertEqual(len(calls), 2)
        self.assertIn("VERIFY", calls[-1][1])

    def test_verified_real_error_remains_rejected(self):
        class Client:
            def chat(self, *args, **kwargs):
                return json.dumps(verdict(False, "major"))

        result = OllamaReviewer(Client()).review("source", "bad translation", "context")
        self.assertFalse(result["ok"])
        self.assertEqual(result["severity"], "major")

    def test_revision_long_paragraph_is_single_complete_request(self):
        class Client:
            model = "configured-model"

            def chat(self, system, user):
                self.payload = (system, user)
                self.calls = getattr(self, "calls", 0) + 1
                return "完整译文"

        client = Client()
        config = Config(chunk_size=100)
        translator = OllamaTranslator(client, config)
        source = "A long source. " * 350 + "SOURCE_END"
        initial = "初译" * 1000 + "INITIAL_END"
        previous = "修订" * 1000 + "PREVIOUS_END"
        history = [{"round": i, **verdict()} for i in range(5)]
        history[-1]["suggestion"] += "FINAL_FIX"
        result = translator.revise(Segment("s", "c", source, "c.xhtml", 0), "en", "context",
                                   initial, previous, history)
        self.assertEqual(result, "完整译文")
        self.assertEqual(client.calls, 1)
        for value in (source, initial, previous, "FINAL_FIX"):
            self.assertIn(value, client.payload[1])

    def test_selection_compares_complete_texts_without_stale_feedback(self):
        calls = []

        class Client:
            def chat(self, system, user, **kwargs):
                calls.append(user)
                return '{"selected_index": 1, "reason": "better"}'

        candidates = ["中" * 4000 + f"END_{i}" for i in range(3)]
        result = OllamaReviewer(Client()).select_best("source", "context", candidates, "STALE_CLAIM")
        self.assertEqual(result["selected_index"], 2)
        self.assertEqual(len(calls), 2)
        for text in candidates:
            self.assertTrue(any(text in call for call in calls))
        self.assertTrue(all("STALE_CLAIM" not in call for call in calls))

    def run_review(self, root, reviewer, translator, *, rounds=2, text="Source text."):
        pipeline = Pipeline(Config(review_model="reviewer", translation_model="translator",
                                   minor_revision_rounds=rounds, serious_revision_rounds=rounds))
        paths = pipeline._paths(root)
        state = JobState("test", "input", "hash", "epub", str(root), source_language="en")
        segment = Segment("s", "c", text, "c.xhtml", 0)
        book = EpubBook(root / "book.epub", "content.opf", [], "Book", [segment])
        translations_path = paths["translations"] / "segments.json"
        translations = json.loads(translations_path.read_text()) if translations_path.exists() else {"s": "初译 bad"}
        with patch("bookflow.pipeline.OllamaReviewer", return_value=reviewer), \
                patch("bookflow.pipeline.OllamaTranslator", return_value=translator):
            pipeline._review(state, book, translations, paths, root / "state.json", "context")
        return json.loads((paths["reviews"] / "review_checkpoint.json").read_text())["llm"][0]

    def test_false_selection_claim_does_not_become_pass_and_resume_is_bounded(self):
        class Reviewer:
            calls = []

            def review(self, source, translation, context):
                self.calls.append((translation, context))
                return verdict()

            def select_best(self, *args):
                return {"selected_index": 1, "reason": "all fixed"}

        class Translator:
            def revise(self, *args, **kwargs):
                return "改后 bad"

        with TemporaryDirectory() as directory:
            root = Path(directory)
            reviewer = Reviewer()
            result = self.run_review(root, reviewer, Translator())
            self.assertFalse(result["ok"])
            self.assertFalse(result["selected_review"]["ok"])
            self.assertEqual(reviewer.calls[-1][0], "改后 bad")
            self.assertTrue(all(context == "context" for _, context in reviewer.calls))
            audit = json.loads(Path(result["audit_file"]).read_text())
            self.assertEqual(audit["candidates"], ["初译 bad", "改后 bad"])
            self.assertEqual(audit["initial_translation"], "初译 bad")
            self.assertEqual(audit["history"][-1]["round"], 2)
            count = len(reviewer.calls)
            self.run_review(root, reviewer, Translator())
            self.assertEqual(len(reviewer.calls), count)

    def test_interrupted_revision_resumes_at_remaining_round(self):
        class Reviewer:
            def review(self, *args):
                return verdict()

            def select_best(self, *args):
                return {"selected_index": 1, "reason": "better"}

        class Translator:
            calls = 0
            rewrite_flags = []

            def revise(self, *args, **kwargs):
                self.calls += 1
                self.rewrite_flags.append(kwargs["rewrite"])
                if self.calls == 2:
                    raise RuntimeError("interrupted")
                return "修订 bad"

        with TemporaryDirectory() as directory:
            root = Path(directory)
            translator = Translator()
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.run_review(root, Reviewer(), translator)
            result = self.run_review(root, Reviewer(), translator)
            audit = json.loads(Path(result["audit_file"]).read_text())
            self.assertEqual([entry["round"] for entry in audit["history"]], [0, 1, 2])
            self.assertEqual(translator.calls, 3)  # one failed request, two successful rounds
            self.assertTrue(translator.rewrite_flags[-1])

    def test_model_cannot_approve_missing_footnote(self):
        class Reviewer:
            def review(self, *args):
                return verdict(True)

            def select_best(self, *args):
                return {"selected_index": 0, "reason": "good"}

        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(Exception, "footnote"):
                self.run_review(Path(directory), Reviewer(), object(), rounds=0,
                                text="Source [[BOOKFLOW_FOOTNOTE_1]]")
            result = json.loads((Path(directory) / "reviews/review_checkpoint.json").read_text())["llm"][0]
            self.assertFalse(result["ok"])
            self.assertIn("footnote", result["issues"][-1])

    def test_fixed_major_does_not_spend_five_rounds_on_remaining_minor(self):
        class Reviewer:
            calls = 0

            def review(self, *args):
                self.calls += 1
                return verdict(False, "major" if self.calls == 1 else "minor")

            def select_best(self, *args):
                return {"selected_index": 1, "reason": "better"}

        class Translator:
            calls = 0

            def revise(self, *args, **kwargs):
                self.calls += 1
                return f"候选{self.calls} bad"

        with TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = Pipeline(Config(review_model="reviewer", translation_model="translator",
                                       minor_revision_rounds=2, serious_revision_rounds=5))
            paths = pipeline._paths(root)
            state = JobState("test", "input", "hash", "epub", str(root), source_language="en")
            segment = Segment("s", "c", "Source text.", "c.xhtml", 0)
            book = EpubBook(root / "book.epub", "content.opf", [], "Book", [segment])
            translator = Translator()
            with patch("bookflow.pipeline.OllamaReviewer", return_value=Reviewer()), \
                    patch("bookflow.pipeline.OllamaTranslator", return_value=translator):
                pipeline._review(state, book, {"s": "初译 bad"}, paths, root / "state.json", "context")
            self.assertEqual(translator.calls, 2)
