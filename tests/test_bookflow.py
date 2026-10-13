from __future__ import annotations

import json
import unittest
import urllib.error
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bookflow.adapters import OllamaClient
from bookflow.config import Config
from bookflow.epub import build_translated_epub, inspect_epub, markdown_to_epub, validate_epub
from bookflow.errors import BookFlowError
from bookflow.inspect import inspect_input
from bookflow.pipeline import Pipeline


def make_epub(path: Path, *, chinese: bool = False) -> None:
    first = "这是一本中文书。这里有第二个段落。" if chinese else "The first chapter has a number. It is 2026."
    second = "这是第二章。" if chinese else "The second chapter keeps the context."
    opf = """<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" unique-identifier="id" version="3.0"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="id">test</dc:identifier><dc:title>Test Book</dc:title><dc:language>en</dc:language></metadata><manifest><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/><item id="c2" href="c2.xhtml" media-type="application/xhtml+xml"/><item id="img" href="image.png" media-type="image/png"/></manifest><spine><itemref idref="c1"/><itemref idref="c2"/></spine></package>"""
    container = """<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"""
    xhtml = lambda text: f"<html xmlns=\"http://www.w3.org/1999/xhtml\"><body><h1>Chapter</h1><p>{text}</p></body></html>"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/c1.xhtml", xhtml(first))
        zf.writestr("OEBPS/c2.xhtml", xhtml(second))
        zf.writestr("OEBPS/style.css", "body { color: black; }")
        zf.writestr("OEBPS/image.png", b"not really an image, but preserved")


def make_footnote_epub(path: Path) -> None:
    opf = """<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" unique-identifier="id" version="3.0"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="id">footnote-test</dc:identifier><dc:title>Footnote Test</dc:title><dc:language>en</dc:language></metadata><manifest><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/><item id="notes" href="notes.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c1"/><itemref idref="notes"/></spine></package>"""
    container = """<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"""
    chapter = """<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Chapter</h1><p>Millions read his work.<a href="notes.xhtml#c01-nts1" id="c01-nts1a"><sup>1</sup></a> They do.</p></body></html>"""
    notes = """<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Notes</h1><p id="c01-nts1"><a href="c1.xhtml#c01-nts1a"><span>1. </span></a>Footnote body.</p></body></html>"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/c1.xhtml", chapter)
        zf.writestr("OEBPS/notes.xhtml", notes)


class FakeOllamaClient:
    calls = 0
    fail_on_call = 0

    def __init__(self, *_args, **_kwargs):
        self.model = _args[1] if len(_args) > 1 else "test-model"

    def chat(self, _system, user, **_kwargs):
        type(self).calls += 1
        if self.fail_on_call and self.calls == self.fail_on_call:
            raise BookFlowError("simulated interruption")
        if '"terms":[' in user:
            return '{"terms": [], "summary": "Source passage facts."}'
        if '"style":{' in user:
            return '{"style":{"genre":"nonfiction","audience":"general","tone":"clear","rules":"faithful"}}'
        original = user.split("Paragraph to translate:\n", 1)[-1]
        return f"译文：{original}"


class BookFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / "book.epub"
        make_epub(self.input)
        FakeOllamaClient.calls = 0
        FakeOllamaClient.fail_on_call = 0

    def tearDown(self):
        self.temp.cleanup()

    def config(self, **kwargs):
        values = dict(
            workspace=self.root / "workspace",
            ollama_url="http://127.0.0.1:11434",
            translation_model="test-model",
            review_enabled=False,
            output_formats=["epub"],
            consistency_enabled=False,
            epubcheck_command="",
        )
        values.update(kwargs)
        return Config(**values)

    def test_inspect_and_epub_roundtrip_preserves_assets(self):
        info = inspect_input(self.input)
        self.assertEqual(info.kind, "epub")
        book = inspect_epub(self.input)
        self.assertEqual(len(book.segments), 4)
        self.assertEqual(validate_epub(self.input), [])

    def test_end_to_end_bilingual_and_cache(self):
        with patch("bookflow.pipeline.OllamaClient", FakeOllamaClient):
            state = Pipeline(self.config()).start(self.input)
        self.assertEqual(state.status, "completed")
        output = Path(state.outputs[0])
        self.assertTrue(output.exists())
        self.assertEqual(FakeOllamaClient.calls, 3)
        with zipfile.ZipFile(output) as zf:
            content = zf.read("OEBPS/c1.xhtml").decode()
            self.assertIn("The first chapter", content)
            self.assertIn("译文：The first chapter", content)
            self.assertLess(content.index("The first chapter"), content.index("译文：The first chapter"))
            self.assertIn("OEBPS/image.png", zf.namelist())
        with patch("bookflow.pipeline.OllamaClient", FakeOllamaClient):
            state2 = Pipeline(self.config()).resume(state.job_id)
        self.assertEqual(state2.status, "completed")
        self.assertEqual(FakeOllamaClient.calls, 3)

    def test_consistency_memory_persists_terms_and_chapter_summaries(self):
        with patch("bookflow.pipeline.OllamaClient", FakeOllamaClient):
            state = Pipeline(self.config(consistency_enabled=True)).start(self.input)
        memory_path = Path(state.workdir) / "metadata" / "translation_memory.json"
        self.assertTrue(memory_path.exists())
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        self.assertTrue(memory["terms_extracted"])
        self.assertEqual(set(memory["chapter_summaries"]), {"chapter-1", "chapter-2"})

    def test_resume_does_not_repeat_completed_segment(self):
        FakeOllamaClient.fail_on_call = 2
        with patch("bookflow.pipeline.OllamaClient", FakeOllamaClient):
            with self.assertRaises(BookFlowError):
                Pipeline(self.config()).start(self.input)
        self.assertEqual(FakeOllamaClient.calls, 2)
        FakeOllamaClient.fail_on_call = 0
        job_id = next(p.name for p in (self.root / "workspace").iterdir())
        with patch("bookflow.pipeline.OllamaClient", FakeOllamaClient):
            state = Pipeline(self.config()).resume(job_id)
        self.assertEqual(state.status, "completed")
        self.assertEqual(FakeOllamaClient.calls, 4)

    def test_chinese_book_skips_translation(self):
        chinese = self.root / "中文.epub"
        make_epub(chinese, chinese=True)
        state = Pipeline(self.config()).start(chinese)
        self.assertTrue(state.skipped_translation)
        self.assertEqual(FakeOllamaClient.calls, 0)

    def test_marker_markdown_is_rendered_as_xhtml(self):
        markdown = self.root / "marker.md"
        image = self.root / "cover.png"
        image.write_bytes(b"fake-png")
        markdown.write_text(
            "# Heading\n\nThis is **bold**, *italic*, `code`, and [a link](https://example.com).\n\n![Cover](cover.png)\n",
            encoding="utf-8",
        )
        output = self.root / "rendered.epub"
        markdown_to_epub(markdown, output, "Rendered")
        with zipfile.ZipFile(output) as zf:
            content = zf.read("OEBPS/chapter.xhtml").decode()
            self.assertIn("<strong>bold</strong>", content)
            self.assertIn("<em>italic</em>", content)
            self.assertIn('<a href="https://example.com">a link</a>', content)
            self.assertIn('<img src="images/cover.png"', content)
            self.assertNotIn("**bold**", content)
            self.assertIn("OEBPS/images/cover.png", zf.namelist())
        self.assertEqual(validate_epub(output), [])

    def test_marker_toc_preserves_hierarchy_and_page_position(self):
        markdown = self.root / "toc-marker.md"
        markdown.write_text(
            "# Book\n\n"
            "### **TABLE OF CONTENTS**\n\n"
            "#### **[Licensing](#page-7-0)**\n\n"
            "#### **[1: Introduction](#page-8-0)**\n\n"
            "[1.1: Overview](#page-9-0) [1.2: Case Study](#page-11-0)\n\n"
            "## <span id=\"page-7-0\"></span>**Licensing**\n\n"
            "License details.\n\n"
            "## <span id=\"page-8-0\"></span>Chapter\n\n"
            "## <span id=\"page-9-0\"></span>Overview\n",
            encoding="utf-8",
        )
        output = self.root / "toc-rendered.epub"
        markdown_to_epub(markdown, output, "Book")
        with zipfile.ZipFile(output) as zf:
            content = zf.read("OEBPS/chapter.xhtml").decode()
            nav = zf.read("OEBPS/nav.xhtml").decode()
            toc_end = content.index("</div>")
            page_break = content.index('<div id="page-7-0" class="bookflow-page-break">')
            self.assertIn('<div id="table-of-contents" class="bookflow-toc">', content)
            self.assertIn('<ul class="toc-sections">', content)
            self.assertIn('<li><a href="#page-9-0">1.1: Overview</a></li>', content)
            self.assertLess(toc_end, page_break)
            self.assertIn('<h2 id="chapter-2"><strong>Licensing</strong></h2>', content)
            self.assertIn('chapter.xhtml#page-7-0', nav)
        self.assertEqual(validate_epub(output), [])

    def test_translation_markdown_is_rendered_as_xhtml(self):
        book = inspect_epub(self.input)
        translations = {
            segment.segment_id: "**粗体**、*斜体*、[链接](https://example.com)"
            for segment in book.segments
        }
        output = self.root / "translated.epub"
        build_translated_epub(self.input, output, translations, mode="bilingual")
        with zipfile.ZipFile(output) as zf:
            content = zf.read("OEBPS/c1.xhtml").decode()
            self.assertIn("<strong>粗体</strong>", content)
            self.assertIn("<em>斜体</em>", content)
            self.assertIn('<a href="https://example.com">链接</a>', content)
            self.assertNotIn("**粗体**", content)
            self.assertLess(
                content.index("<p>The first chapter"),
                content.index('<p class="bookflow-translation"><strong>粗体</strong>'),
            )
        self.assertEqual(validate_epub(output), [])

    def test_footnote_markers_are_protected_and_restored(self):
        source = self.root / "footnote.epub"
        make_footnote_epub(source)
        book = inspect_epub(source)
        main = next(segment for segment in book.segments if "Millions" in segment.text)
        note = next(segment for segment in book.segments if "Footnote body" in segment.text)
        self.assertIn("[[BOOKFLOW_FOOTNOTE_1]]", main.text)
        self.assertNotIn("1.", main.text)
        self.assertIn("[[BOOKFLOW_FOOTNOTE_1]]", note.text)
        self.assertNotIn("1.", note.text)

        translations = {segment.segment_id: f"译文：{segment.text}" for segment in book.segments}
        output = self.root / "footnote-translated.epub"
        build_translated_epub(source, output, translations, mode="bilingual")
        with zipfile.ZipFile(output) as zf:
            chapter = zf.read("OEBPS/c1.xhtml").decode()
            notes = zf.read("OEBPS/notes.xhtml").decode()
            self.assertIn('href="notes.xhtml#c01-nts1"', chapter)
            self.assertIn('href="c1.xhtml#c01-nts1a"', notes)
            self.assertNotIn("[[BOOKFLOW_FOOTNOTE_1]]", chapter + notes)
            self.assertIn("<sup>1</sup>", chapter)
            self.assertIn("Footnote body.", notes)
            self.assertEqual(chapter.count("Millions read his work."), 2)
            self.assertEqual(chapter.count("They do."), 2)
            self.assertEqual(notes.count("Footnote body."), 2)
        self.assertEqual(validate_epub(output), [])

    def test_original_mode_requires_no_model(self):
        state = Pipeline(self.config(translation_model="", output_mode="original")).start(self.input)
        self.assertEqual(state.status, "completed")
        self.assertEqual(FakeOllamaClient.calls, 0)

    def test_review_revision_is_accepted_only_after_post_review_pass(self):
        class ReviewLoopClient:
            pass_after_revision = True
            revision_prompts = []

            def __init__(self, *_args, **_kwargs):
                self.model = _args[1] if len(_args) > 1 else "test-model"

            def chat(self, _system, user, **kwargs):
                if '"terms":[' in user:
                    return '{"terms": [], "summary": "Source passage facts."}'
                if '"style":{' in user:
                    return '{"style":{"genre":"nonfiction","audience":"general","tone":"clear","rules":"faithful"}}'
                if "json_format" in kwargs:
                    if "Original:\nThe first chapter" in user and "修订译文" not in user:
                        return '{"ok": false, "issues": ["需要修订措辞"]}'
                    if "Original:\nThe first chapter" in user and not self.pass_after_revision:
                        return '{"ok": false, "issues": ["修订后仍有问题"]}'
                    return '{"ok": true, "issues": []}'
                if "Current correction request" in user:
                    type(self).revision_prompts.append(user)
                source = user.split("Paragraph to translate:\n", 1)[-1]
                if "Original (the only text to translate):\n" in user:
                    source = user.split("Original (the only text to translate):\n", 1)[-1]
                prefix = "修订译文：" if "Current correction request" in user else "译文："
                return prefix + source

        with patch("bookflow.pipeline.OllamaClient", ReviewLoopClient), patch("bookflow.pipeline.Pipeline._validate_artifact"):
            state = Pipeline(
                self.config(
                    review_enabled=True,
                    review_model="reviewer",
                    max_revision_rounds=2,
                    consistency_enabled=True,
                    strict_review=True,
                )
            ).start(self.input)
        self.assertEqual(state.status, "completed")
        translations = json.loads((Path(state.workdir) / "translations" / "segments.json").read_text(encoding="utf-8"))
        paragraph = next(segment for segment in inspect_epub(self.input).segments if segment.text.startswith("The first chapter"))
        self.assertTrue(translations[paragraph.segment_id].startswith("修订译文："))
        self.assertTrue(ReviewLoopClient.revision_prompts)
        revision_prompt = ReviewLoopClient.revision_prompts[0]
        for marker in (
            "Initial translation (round 0)",
            "Initial review (round 0)",
            "Previous candidate translation",
            "Complete review feedback history",
            "completely reorganize",
        ):
            self.assertIn(marker, revision_prompt)

    def test_review_rejected_revision_is_not_written_in_strict_mode(self):
        class AlwaysRejectClient:
            def __init__(self, *_args, **_kwargs):
                self.model = _args[1] if len(_args) > 1 else "test-model"

            def chat(self, _system, user, **kwargs):
                if "json_format" in kwargs and "Original:\nThe first chapter" in user:
                    return '{"ok": false, "issues": ["修订后仍有问题"]}'
                if "json_format" in kwargs:
                    return '{"ok": true, "issues": []}'
                source = user.split("Paragraph to translate:\n", 1)[-1]
                return ("修订译文：" if "Current correction request" in user else "译文：") + source

        with patch("bookflow.pipeline.OllamaClient", AlwaysRejectClient):
            with self.assertRaises(BookFlowError):
                Pipeline(
                    self.config(
                        review_enabled=True,
                        review_model="reviewer",
                        max_revision_rounds=1,
                        strict_review=True,
                    )
                ).start(self.input)
        job_id = next(path.name for path in (self.root / "workspace").iterdir())
        translations = json.loads(
            (self.root / "workspace" / job_id / "translations" / "segments.json").read_text(encoding="utf-8")
        )
        paragraph = next(segment for segment in inspect_epub(self.input).segments if segment.text.startswith("The first chapter"))
        self.assertTrue(translations[paragraph.segment_id].startswith("译文："))
        self.assertTrue((self.root / "workspace" / job_id / "reviews" / "review_checkpoint.json").exists())

    def test_ollama_payload_uses_chat_endpoint_and_options(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"message":{"content":"ok"}}'

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode())
            captured["timeout"] = timeout
            return Response()

        result = OllamaClient("http://localhost:11434", "demo", opener=opener, options={"temperature": 0.2}).chat("s", "u")
        self.assertEqual(result, "ok")
        self.assertEqual(captured["url"], "http://localhost:11434/api/chat")
        self.assertFalse(captured["payload"]["stream"])
        self.assertEqual(captured["payload"]["options"]["temperature"], 0.2)

    def test_ollama_retries_transient_http_errors(self):
        calls = 0

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"message":{"content":"ok"}}'

        def opener(_request, timeout):
            self.assertEqual(timeout, 600)
            nonlocal calls
            calls += 1
            if calls < 3:
                raise urllib.error.HTTPError("http://localhost:11434/api/chat", 500, "server restarting", {}, None)
            return Response()

        client = OllamaClient(
            "http://localhost:11434",
            "demo",
            opener=opener,
            max_retries=2,
            retry_backoff_seconds=0,
        )
        self.assertEqual(client.chat("s", "u"), "ok")
        self.assertEqual(calls, 3)

    def test_bad_input_is_explicit(self):
        with self.assertRaises(BookFlowError):
            Pipeline(self.config()).start(self.root / "bad.txt")


if __name__ == "__main__":
    unittest.main()
