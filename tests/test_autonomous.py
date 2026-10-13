from __future__ import annotations

import json
import subprocess
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bookflow.adapters import OllamaTranslator, OllamaClient, OllamaReviewer
from bookflow.book_review import chapter_review
from bookflow.config import Config
from bookflow.epub import EpubBook, Segment, inspect_epub, build_translated_epub, markdown_to_epub
from bookflow.errors import BookFlowError, ExternalToolError
from bookflow.memory import TranslationMemory, semantic_chunks, summarize_chapter, term_occurs, valid_term_exception
from bookflow.pipeline import Pipeline, JobState
from bookflow.preparation import prepare_book
from bookflow.qa import deterministic_qa, numeric_advisories, numeric_evidence
from tests.test_bookflow import make_epub


STYLE = {"genre": "nonfiction", "audience": "general", "tone": "precise", "rules": "Preserve meaning."}


class Editor:
    """A protocol fake with persistent outputs and inspectable model inputs."""
    model = "local-test"
    def __init__(self, *args, **kwargs):
        self.calls = []

    def chat(self, system, user, **kwargs):
        self.calls.append((system, user))
        if "Extract recurring" in user:
            terms = [{"source": name, "target": target} for name, target in [("Brier", "布里尔"), ("Alpha", "阿尔法")]
                     if name in user.split("Excerpt:\n")[-1]]
            return json.dumps({"terms": terms, "summary": "Facts from this passage."})
        if "Verify and reconcile" in user:
            return json.dumps({"terms": [{"source": "Brier", "target": "布里尔", "meaning": "score"}]
                               if '"source": "Brier"' in user else [{"source": "Alpha", "target": "阿尔法"}]})
        if "Infer a concise" in user:
            return json.dumps({"style": STYLE})
        if "Review this book passage" in user:
            return '{"issues": []}'
        if "Original:\n" in user and "Translation:\n" in user:
            return '{"ok": true, "severity": "none", "issues": [], "suggestion": ""}'
        if "selected_index" in system:
            return '{"selected_index": 0, "reason": "retained"}'
        if "continuity summary" in user or "Merge these sequential" in user or "Condense these sequential" in user:
            return "A factual summary."
        source = user.split("Paragraph to translate:\n")[-1]
        if "Original (the only text to translate):\n" in user:
            source = user.split("Original (the only text to translate):\n")[-1]
        return source.replace("Brier", "布里尔").replace("Alpha", "阿尔法")


def book_at(root, texts):
    return EpubBook(root / "book.epub", "content.opf", [], "A technical book",
                    [Segment(f"s{i}", f"c{i}", t, "c.xhtml", i) for i, t in enumerate(texts)])


class AutonomousTests(unittest.TestCase):
    def test_reasoning_verifier_only_runs_for_rejections(self):
        fast, deep = Editor(), Editor()
        reviewer = OllamaReviewer(fast, verifier=OllamaReviewer(deep))
        self.assertTrue(reviewer.review("source", "译文", "policy")["ok"])
        self.assertEqual(len(deep.calls), 0)
        rejected = {"ok": False, "severity": "major", "issues": ["alleged omission"], "suggestion": "restore"}
        with patch.object(fast, "chat", return_value=json.dumps(rejected)):
            result = reviewer.review("source", "译文", "policy")
        self.assertTrue(result["ok"])
        self.assertFalse(result["preliminary_review"]["ok"])
        self.assertEqual(len(deep.calls), 1)
        self.assertIn("alleged omission", deep.calls[0][1])
        self.assertIn("policy", deep.calls[0][1])

    def test_unavailable_reasoning_verifier_falls_back_with_audit_record(self):
        fast, deep = Editor(), Editor()
        rejected = {"ok": False, "severity": "major", "issues": ["alleged omission"], "suggestion": "restore"}
        accepted = {"ok": True, "severity": "none", "issues": [], "suggestion": ""}
        with patch.object(fast, "chat", side_effect=[json.dumps(rejected), json.dumps(accepted)]), \
             patch.object(deep, "chat", side_effect=ExternalToolError("reasoning unavailable")):
            result = OllamaReviewer(fast, verifier=OllamaReviewer(deep)).review("source", "译文", "policy")
        self.assertTrue(result["ok"])
        self.assertEqual(result["verification_error"], "reasoning unavailable")
        self.assertFalse(result["preliminary_review"]["ok"])

    def test_chapter_verifier_removes_unsupported_allegations_and_caches(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            book = book_at(root, ["The no campaign won."])
            memory = TranslationMemory(root / "m.json", book.title)
            fast, deep = Editor(), Editor()
            alleged = {"issues": [{"segment_id": "s0", "severity": "major", "source_quote": "no",
                       "translation_quote": "反对", "issue": "optional wording", "suggestion": "replace"}]}
            with patch.object(fast, "chat", return_value=json.dumps(alleged)):
                args = (book, {"s0": "反对方获胜。"}, memory, fast, root / "chapter.json")
                self.assertEqual(chapter_review(*args, chars=500, retries=0, verifier_client=deep), [])
                self.assertEqual(chapter_review(*args, chars=500, retries=0, verifier_client=deep), [])
            self.assertEqual(len(deep.calls), 1)
            saved = next(iter(json.loads((root / "chapter.json").read_text()).values()))
            self.assertEqual(saved["preliminary_issues"], alleged["issues"])
            self.assertEqual(saved["issues"], [])

    def test_whole_book_preparation_and_human_override_resume(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            book = book_at(root, ["Alpha is introduced.", "ordinary words. " * 1500, "Brier measures forecast error."])
            memory = TranslationMemory(root / "memory.json", book.title)
            config = Config(translation_model="local", glossary={"Brier": "布赖尔"}, style={"tone": "formal"})
            client = Editor()
            prepare_book(book, memory, client, config, root / "preparation.json")
            self.assertEqual(memory.terms["Brier"], "布赖尔")
            self.assertEqual(memory.style["tone"], "formal")
            self.assertEqual(memory.style["genre"], "nonfiction")
            self.assertTrue(memory.book_brief)
            self.assertEqual(set(memory.chapter_briefs), {"c0", "c1", "c2"})
            self.assertTrue(any("Brier measures" in call[1] and "Extract recurring" in call[1] for call in client.calls))
            count = len(client.calls)
            config.glossary["Brier"] = "布里尔评分"
            prepare_book(book, memory, client, config, root / "preparation.json")
            self.assertEqual(len(client.calls), count)
            self.assertEqual(memory.terms["Brier"], "布里尔评分")
            self.assertEqual(memory.entries["Brier"]["origin"], "human")

    def test_preparation_failure_is_retryable_not_completed(self):
        class Broken(Editor):
            def chat(self, *args, **kwargs):
                return "bad JSON"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            book = book_at(root, ["Brier score."])
            memory = TranslationMemory(root / "memory.json", book.title)
            config = Config(translation_model="local", request_retries=0)
            with self.assertRaises(BookFlowError):
                prepare_book(book, memory, Broken(), config, root / "preparation.json")
            self.assertFalse(memory.terms_extracted)
            prepare_book(book, memory, Editor(), config, root / "preparation.json")
            self.assertTrue(memory.terms_extracted)

    def test_policy_is_not_clipped_and_matches_whole_words(self):
        memory = TranslationMemory(Path("unused"), "Book", terms={"art": "艺术", "Brier": "布里尔"},
                                   chapter_summaries={"c0": "摘要" * 1000}, style=STYLE)
        self.assertNotIn("art", memory.relevant_terms("party"))
        self.assertTrue(term_occurs("forecaster", "forecasters' scores"))
        memory.record("p", "c1", "最近译文" * 200)
        context = memory.context_for(chapter="c1", source_text="Brier is relevant.")
        client = Editor()
        OllamaTranslator(client, Config()).translate(Segment("s", "c1", "Brier is relevant.", "c", 1), "en", context)
        prompt = client.calls[-1][1]
        self.assertIn("Brier => 布里尔", prompt)
        self.assertIn("最近译文", prompt)
        self.assertIn("Mandatory book style", prompt)
        pipeline = Pipeline(Config())
        memory.book_brief = "WHOLE_BOOK_OVERVIEW"
        memory.chapter_briefs = {"c0": "CURRENT_CHAPTER_OVERVIEW"}
        pipeline.memory = memory
        reference = pipeline._segment_context(book_at(Path("."), ["Brier is relevant."]), 0, "")
        self.assertIn("WHOLE_BOOK_OVERVIEW", reference)
        self.assertIn("CURRENT_CHAPTER_OVERVIEW", reference)

    def test_manual_glossary_is_also_filtered_in_initial_and_revision_requests(self):
        client = Editor()
        config = Config(glossary={"Brier": "布里尔", "UNRELATED_SENTINEL": "无关术语"})
        translator = OllamaTranslator(client, config)
        segment = Segment("s", "c", "Brier measures error.", "c", 1)
        translator.translate(segment, "en", "context")
        self.assertNotIn("UNRELATED_SENTINEL", client.calls[-1][1])
        self.assertIn("Brier", client.calls[-1][1])
        translator.revise(segment, "en", "context", "初稿", "前稿", [{"suggestion": "fix"}])
        self.assertNotIn("UNRELATED_SENTINEL", client.calls[-1][1])

    def test_semantic_chunking_preserves_every_character_and_placeholder(self):
        source = "A sentence about forecasting. " * 12 + "[[BOOKFLOW_FOOTNOTE_1]] " + "中文不能丢失。" * 100
        pieces = semantic_chunks(source, 100)
        self.assertEqual("".join(pieces), source)
        self.assertTrue(all(len(p) <= 100 for p in pieces))
        self.assertEqual(sum("[[BOOKFLOW_FOOTNOTE_1]]" in p for p in pieces), 1)

    def test_full_chapter_summary_includes_late_source(self):
        client = Editor()
        summarize_chapter(client, title="Book", chapter="c1", source_text="Opening. " * 1300 + "LATE_REVELATION", translation_text="旧摘要")
        self.assertTrue(any("LATE_REVELATION" in call[1] for call in client.calls))

    def test_chapter_review_uses_neighbors_and_rechecks_changed_text(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            book = book_at(root, ["The no campaign opposed separation.", "As it turned out, no won."])
            memory = TranslationMemory(root / "m.json", book.title, style=STYLE)
            client = Editor()
            translations = {"s0": "反对独立的一方。", "s1": "反对独立的一方获胜。"}
            args = (book, translations, memory, client, root / "chapter.json")
            self.assertEqual(chapter_review(*args, chars=500, retries=0), [])
            count = len(client.calls)
            self.assertIn("no campaign", client.calls[-1][1])
            chapter_review(*args, chars=500, retries=0)
            self.assertEqual(len(client.calls), count)
            translations["s1"] = "无人获胜。"
            chapter_review(*args, chars=500, retries=0)
            self.assertGreater(len(client.calls), count)

    def test_footnotes_are_hard_checks_even_without_model_review(self):
        segment = Segment("s", "c", "Text [[BOOKFLOW_FOOTNOTE_1]] then [[BOOKFLOW_FOOTNOTE_2]]", "c", 1)
        for translation in ["译文", "译文 [[BOOKFLOW_FOOTNOTE_2]] [[BOOKFLOW_FOOTNOTE_1]]"]:
            self.assertFalse(deterministic_qa([segment], {"s": translation}).ok)
        self.assertTrue(numeric_advisories("10 people", "10人，另有999人"))
        self.assertTrue(deterministic_qa([Segment("s", "c", "100000 pounds", "c", 1)], {"s": "10万英镑"}).ok)

    def test_mixed_source_text_and_inline_images_survive(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.epub"
            make_epub(original)
            source = root / "mixed.epub"
            with zipfile.ZipFile(original) as src, zipfile.ZipFile(source, "w") as dst:
                for item in src.infolist():
                    data = src.read(item.filename)
                    if item.filename == "OEBPS/c1.xhtml":
                        data = b'<html xmlns="http://www.w3.org/1999/xhtml"><body>Loose opening<div>Before<p>Middle<img src="image.png"/></p>After</div><p>I</p></body></html>'
                    dst.writestr(item, data)
            book = inspect_epub(source)
            texts = [s.text for s in book.segments]
            for expected in ["Loose opening", "Before", "After", "I"]:
                self.assertIn(expected, texts)
            image_segment = next(s for s in book.segments if "BOOKFLOW_IMAGE" in s.text)
            import hashlib
            old_id = hashlib.sha256("OEBPS/c1.xhtml\x001\x00Middle".encode()).hexdigest()[:20]
            self.assertEqual(book.legacy_ids[image_segment.segment_id], old_id)
            translations = {s.segment_id: s.text for s in book.segments}
            output = root / "translated.epub"
            build_translated_epub(source, output, translations, "translated")
            self.assertEqual([s.text for s in inspect_epub(output).segments], texts)
            with zipfile.ZipFile(output) as archive:
                self.assertIn('src="image.png"', archive.read("OEBPS/c1.xhtml").decode())
            translations[image_segment.segment_id] = "图片丢失"
            self.assertFalse(deterministic_qa(book.segments, translations).ok)

    def test_end_to_end_acceptance_and_recovery_without_repeated_model_work(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            md = root / "source.md"
            md.write_text("# Forecasting\n\nBrier measures forecast error.\n\nThe rate is 10%.\n")
            source = markdown_to_epub(md, root / "source.epub", "Forecasting")
            config = Config(workspace=root / "work", translation_model="local-test", review_model="local-test", epubcheck_command="test-epubcheck")
            clients = []
            def factory(*args, **kwargs):
                client = Editor()
                clients.append(client)
                return client
            with patch("bookflow.pipeline.OllamaClient", side_effect=factory), \
                 patch("bookflow.pipeline.shutil.which", return_value="test-epubcheck"), \
                 patch("bookflow.adapters.run_command", return_value=subprocess.CompletedProcess([], 0, "passed", "")):
                state = Pipeline(config).start(source)
                self.assertEqual(state.quality_status, "finalized")
                count = sum(len(c.calls) for c in clients)
                state = Pipeline(config).resume(state.job_id)
                self.assertEqual(sum(len(c.calls) for c in clients), count)
                self.assertEqual(state.quality_status, "finalized")
            acceptance = json.loads((Path(state.workdir) / "reviews/acceptance.json").read_text())
            self.assertFalse(acceptance["human_approval"])
            self.assertTrue(acceptance["outputs"][0]["sha256"])

    def test_deterministic_failure_triggers_revision_before_blocking(self):
        class Reviewer:
            def review(self, *args):
                return {"ok": True, "severity": "none", "issues": [], "suggestion": ""}
        class Translator:
            def revise(self, *args, **kwargs):
                return "10人 [[BOOKFLOW_FOOTNOTE_1]]"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = Config(translation_model="local", review_model="local")
            pipeline = Pipeline(config)
            paths = pipeline._paths(root)
            book = book_at(root, ["10 people [[BOOKFLOW_FOOTNOTE_1]]"])
            state = JobState("job", "in", "hash", "epub", str(root), source_language="en")
            translations = {"s0": "9人"}
            with patch("bookflow.pipeline.OllamaReviewer", return_value=Reviewer()), patch("bookflow.pipeline.OllamaTranslator", return_value=Translator()):
                pipeline._review(state, book, translations, paths, root / "state.json", "context")
            self.assertEqual(translations["s0"], "10人 [[BOOKFLOW_FOOTNOTE_1]]")

    def test_human_term_cannot_be_waived_by_model(self):
        class Reviewer:
            def review(self, *args):
                return {"ok": True, "severity": "none", "issues": [], "suggestion": "",
                        "terminology_exceptions": {"Brier": "optional synonym"}}
            def select_best(self, *args):
                return {"selected_index": 0, "reason": "best available"}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = Pipeline(Config(translation_model="local", minor_revision_rounds=0, serious_revision_rounds=0))
            pipeline.memory = TranslationMemory(root / "memory.json", "Book", terms={"Brier": "布里尔"},
                                               entries={"Brier": {"origin": "human"}})
            paths = pipeline._paths(root)
            state = JobState("job", "in", "hash", "epub", str(root), source_language="en")
            with patch("bookflow.pipeline.OllamaReviewer", return_value=Reviewer()):
                pipeline._review(state, book_at(root, ["Brier score"]), {"s0": "布赖尔评分"}, paths, root / "state.json", "context")
            review = json.loads((paths["reviews"] / "quality.json").read_text())
            self.assertFalse(review["llm"][0]["ok"])

    def test_changing_one_term_revisits_only_affected_segment(self):
        class Reviewer:
            def __init__(self):
                self.calls = []
            def review(self, source, translated, context):
                self.calls.append(source)
                return {"ok": True, "severity": "none", "issues": [], "suggestion": ""}
        class Translator:
            def revise(self, *args, **kwargs):
                return "布赖尔评分"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = Pipeline(Config(translation_model="local"))
            memory = TranslationMemory(root / "memory.json", "Book", terms={"Brier": "布里尔"})
            pipeline.memory = memory
            book = book_at(root, ["Brier score", "An unrelated paragraph"])
            paths = pipeline._paths(root)
            state = JobState("job", "in", "hash", "epub", str(root), source_language="en")
            translations = {"s0": "布里尔评分", "s1": "无关段落"}
            reviewer = Reviewer()
            with patch("bookflow.pipeline.OllamaReviewer", return_value=reviewer), patch("bookflow.pipeline.OllamaTranslator", return_value=Translator()):
                pipeline.review_contexts = {s.segment_id: pipeline._segment_context(book, i, "context") for i, s in enumerate(book.segments)}
                pipeline._review(state, book, translations, paths, root / "state.json", "context")
                self.assertEqual(len(reviewer.calls), 2)
                memory.terms["Brier"] = "布赖尔"
                pipeline.review_contexts = {s.segment_id: pipeline._segment_context(book, i, "context") for i, s in enumerate(book.segments)}
                pipeline._review(state, book, translations, paths, root / "state.json", "context")
            self.assertEqual(reviewer.calls.count("An unrelated paragraph"), 1)
            self.assertEqual(translations["s0"], "布赖尔评分")

    def test_chapter_correction_refreshes_summary_and_resume_does_not_repeat(self):
        class ContextEditor(Editor):
            def chat(self, system, user, **kwargs):
                if "Review this book passage" in user and '"translation": "no won"' in user:
                    records = json.loads(user.split("Records:\n")[-1])
                    row = next(r for r in records if r["original"] == "no won")
                    return json.dumps({"issues": [{"segment_id": row["segment_id"], "severity": "major",
                        "source_quote": "no won", "translation_quote": "no won", "issue": "unclear referendum outcome", "suggestion": "Use 反对方获胜"}]})
                if "Original:\nno won\n" in user and "Chapter-level allegations" in user and "Translation:\n反对方获胜" not in user:
                    return '{"ok":false,"severity":"major","issues":["wrong referendum outcome"],"suggestion":"Use 反对方获胜"}'
                if "Original (the only text to translate):\nno won" in user:
                    return "反对方获胜"
                return super().chat(system, user, **kwargs)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            md = root / "source.md"
            md.write_text("# Vote\n\nThe no campaign opposed separation.\n\nno won\n")
            source = markdown_to_epub(md, root / "source.epub", "Vote")
            config = Config(workspace=root / "work", translation_model="local", epubcheck_command="")
            clients = []
            def factory(*args, **kwargs):
                client = ContextEditor(); clients.append(client); return client
            with patch("bookflow.pipeline.OllamaClient", side_effect=factory):
                state = Pipeline(config).start(source)
                work = Path(state.workdir)
                translations = json.loads((work / "translations/segments.json").read_text())
                self.assertIn("反对方获胜", translations.values())
                checkpoint = json.loads((work / "reviews/book_pass.json").read_text())
                self.assertEqual(checkpoint["round"], 1)
                self.assertEqual(checkpoint["issues"], [])
                calls = sum(len(c.calls) for c in clients)
                Pipeline(config).resume(state.job_id)
                self.assertEqual(sum(len(c.calls) for c in clients), calls)

    def test_overlapping_human_term_takes_priority_over_auto_term(self):
        memory = TranslationMemory(Path("unused"), "Book", terms={"Brier": "布赖尔", "Brier score": "布里尔评分"},
                                   entries={"Brier": {"origin": "local_model"}, "Brier score": {"origin": "human"}})
        self.assertEqual(memory.relevant_terms("A Brier score"), {"Brier score": "布里尔评分"})
        self.assertEqual(memory.term_violations("A Brier score", "一个布里尔评分"), {})

    def test_no_review_cannot_be_finalized(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "book.epub"
            make_epub(source)
            config = Config(workspace=root / "work", translation_model="local", review_enabled=False,
                            consistency_enabled=False, epubcheck_command="")
            with patch("bookflow.pipeline.OllamaClient", Editor):
                state = Pipeline(config).start(source)
            self.assertEqual(state.quality_status, "draft")
            self.assertTrue(Path(state.outputs[0] + ".quality.json").exists())

    def test_numeric_evidence_preserves_currency_and_equal_magnitude(self):
        facts = numeric_evidence("100,000 pounds; 1.3 billion dollars", "10 万英镑；13 亿美元")
        for source, target in zip(facts["source"], facts["translation"]):
            self.assertEqual(source["magnitudes"], target["magnitudes"])
            self.assertEqual(source["currency"], target["currency"])
        self.assertEqual(facts["source"][1]["magnitudes"], ["1300000000"])

    def test_automatic_term_exception_requires_actual_source_evidence(self):
        self.assertFalse(valid_term_exception("an acceptable synonym", "resolution"))
        self.assertFalse(valid_term_exception({"kind": "different_sense", "reason": "different", "source_quote": "invented"}, "resolution"))
        self.assertTrue(valid_term_exception({"kind": "different_sense", "reason": "a formal motion", "source_quote": "resolution"}, "a resolution was passed"))

    def test_stalled_chapter_issue_remains_draft_and_is_bounded_on_resume(self):
        class Stalled(Editor):
            def chat(self, system, user, **kwargs):
                if "Review this book passage" in user:
                    self.calls.append((system, user))
                    row = json.loads(user.split("Records:\n")[-1])[0]
                    return json.dumps({"issues": [{"segment_id": row["segment_id"], "severity": "minor",
                        "source_quote": row["original"], "translation_quote": row["translation"],
                        "issue": "persistent disputed style issue", "suggestion": "adjust style"}]})
                return super().chat(system, user, **kwargs)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            md = root / "source.md"; md.write_text("# Vote\n\nThe vote ended.\n")
            source = markdown_to_epub(md, root / "source.epub", "Vote")
            config = Config(workspace=root / "work", translation_model="local", epubcheck_command="", book_revision_rounds=2)
            clients = []
            def factory(*args, **kwargs):
                client = Stalled(); clients.append(client); return client
            with patch("bookflow.pipeline.OllamaClient", side_effect=factory):
                state = Pipeline(config).start(source)
                self.assertEqual(state.quality_status, "draft")
                self.assertEqual(state.quality_summary["chapter_issues"], 1)
                count = sum(len(c.calls) for c in clients)
                Pipeline(config).resume(state.job_id)
                self.assertEqual(sum(len(c.calls) for c in clients), count)

    def test_complete_request_expands_budget_without_clipping(self):
        payloads = []
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"message":{"content":"done"}}'
        def opener(request, **kwargs):
            payloads.append(json.loads(request.data))
            return Response()
        client = OllamaClient("http://localhost:11434", "local", opener=opener, options={"num_ctx": 4096})
        source = "正文" * 2500
        self.assertEqual(client.chat("rules", source), "done")
        self.assertEqual(payloads[0]["messages"][-1]["content"], source)
        self.assertGreater(payloads[0]["options"]["num_ctx"], 4096)
        self.assertEqual(client.options["num_ctx"], 4096)
        with self.assertRaisesRegex(BookFlowError, "上下文预算"):
            client.chat("rules", "文" * 20000)
        self.assertEqual(len(payloads), 1)

    def test_empty_ocr_does_not_become_a_successful_fake_book(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            md = root / "empty.md"; md.write_text("")
            with self.assertRaisesRegex(BookFlowError, "未提取"):
                markdown_to_epub(md, root / "empty.epub")

    def test_repeated_unchanged_revisions_stop_early_without_false_acceptance(self):
        class Reviewer:
            def review(self, *args):
                return {"ok": False, "severity": "major", "issues": ["persistent issue"], "suggestion": "fix"}
            def select_best(self, *args):
                return {"selected_index": 0, "reason": "best available"}
        class Translator:
            calls = 0
            def revise(self, *args, **kwargs):
                self.calls += 1
                return "unchanged"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = Pipeline(Config(translation_model="local", serious_revision_rounds=5))
            paths = pipeline._paths(root)
            translator = Translator()
            state = JobState("job", "in", "hash", "epub", str(root), source_language="en")
            with patch("bookflow.pipeline.OllamaReviewer", return_value=Reviewer()), patch("bookflow.pipeline.OllamaTranslator", return_value=translator):
                pipeline._review(state, book_at(root, ["source"]), {"s0": "unchanged"}, paths, root / "state.json", "context")
            self.assertEqual(translator.calls, 2)
            report = json.loads((paths["reviews"] / "quality.json").read_text())
            self.assertFalse(report["llm"][0]["ok"])


if __name__ == "__main__":
    unittest.main()
