from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bookflow.adapters import BabelEbookAdapter, CalibreAdapter, MarkerAdapter, OllamaReviewer, OllamaTranslator
from bookflow.config import Config
from bookflow.epub import Segment


class AdapterContractTests(unittest.TestCase):
    def test_translategemma_heading_and_residual_retry(self):
        class FakeClient:
            model = "translategemma:12b"

            def __init__(self):
                self.calls = []

            def chat(self, system, user, **_kwargs):
                self.calls.append((system, user))
                return "含有 tireless" if len(self.calls) == 1 else "不知疲倦的"

        client = FakeClient()
        translator = OllamaTranslator(client, Config(translation_model=client.model, target_language="zh-CN"))
        heading = Segment("h", "chapter-8", "Chapter 1", "chapter.xhtml", 1)
        paragraph = Segment("p", "chapter-8", "A tireless worker.", "chapter.xhtml", 2)
        self.assertEqual(translator.translate(heading, "en", "book context"), "第1章")
        self.assertEqual(translator.translate(paragraph, "en", "book context"), "不知疲倦的")
        self.assertEqual(len(client.calls), 2)
        self.assertIn("tireless", client.calls[1][1])

    def test_ollama_prompt_protects_footnote_placeholders(self):
        class FakeClient:
            model = "qwen3.5:latest"

            def __init__(self):
                self.calls = []

            def chat(self, system, user, **_kwargs):
                self.calls.append((system, user))
                return "译文 [[BOOKFLOW_FOOTNOTE_1]]"

        client = FakeClient()
        translator = OllamaTranslator(client, Config(translation_model=client.model, target_language="zh-CN"))
        segment = Segment("p", "chapter-1", "正文 [[BOOKFLOW_FOOTNOTE_1]] 后文", "chapter.xhtml", 1)
        result = translator.translate(segment, "en", "book context")
        self.assertIn("[[BOOKFLOW_FOOTNOTE_1]]", result)
        self.assertIn("placeholder", client.calls[0][0].lower())
        self.assertIn("[[BOOKFLOW_FOOTNOTE_1]]", client.calls[0][1])

    def test_reviewer_retries_invalid_json_and_accepts_valid_result(self):
        class FakeClient:
            def __init__(self):
                self.calls = 0

            def chat(self, _system, _user, **_kwargs):
                self.calls += 1
                return "not json" if self.calls == 1 else '{"ok": true, "issues": [], "suggestion": ""}'

        client = FakeClient()
        result = OllamaReviewer(client, max_parse_retries=1).review("原文", "译文", "上下文")
        self.assertEqual(result, {"ok": True, "severity": "none", "issues": [], "suggestion": ""})
        self.assertEqual(client.calls, 2)

    def test_reviewer_exhausted_invalid_json_is_not_ok_and_not_revision_text(self):
        class FakeClient:
            def chat(self, _system, _user, **_kwargs):
                return '{"ok": false, "issues": ["截断"]'

        result = OllamaReviewer(FakeClient(), max_parse_retries=1).review("原文", "译文", "上下文")
        self.assertFalse(result["ok"])
        self.assertEqual(result["issues"], ["审校模型未返回合法 JSON"])
        self.assertEqual(result["suggestion"], "")
        self.assertEqual(result["review_error"], "invalid_json")

    def test_reviewer_rejects_ok_true_with_issues(self):
        class FakeClient:
            def __init__(self):
                self.calls = 0

            def chat(self, _system, _user, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    return '{"ok": true, "issues": ["有问题"], "suggestion": "请修正"}'
                return '{"ok": true, "issues": [], "suggestion": ""}'

        client = FakeClient()
        result = OllamaReviewer(client, max_parse_retries=1).review("原文", "译文", "上下文")
        self.assertTrue(result["ok"])
        self.assertEqual(result["issues"], [])
        self.assertEqual(client.calls, 2)

    def test_reviewer_selects_best_candidate(self):
        class FakeClient:
            def chat(self, _system, _user, **kwargs):
                self.assertIn("selected_index", kwargs["json_format"]["properties"])
                return '{"selected_index": 1, "reason": "候选二更完整"}'

            def assertIn(self, *args):
                # Keep this fake independent of unittest.TestCase.
                if args[0] not in args[1]:
                    raise AssertionError(args)

        result = OllamaReviewer(FakeClient(), max_parse_retries=0).select_best(
            "原文", "上下文", ["候选一", "候选二"], "Round 0: issue"
        )
        self.assertEqual(result["selected_index"], 1)
        self.assertEqual(result["reason"], "候选二更完整")

    def test_marker_cli_contract_includes_ocr_switch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "marker"

            def fake_run(args, **_kwargs):
                output.mkdir(parents=True, exist_ok=True)
                (output / "book.md").write_text("# Title", encoding="utf-8")
                return subprocess.CompletedProcess(args, 0, "", "")

            with patch("bookflow.adapters.run_command", side_effect=fake_run) as run:
                result = MarkerAdapter("marker_single").convert(root / "book.pdf", output, "force")
            self.assertEqual(result.name, "book.md")
            args = run.call_args.args[0]
            self.assertIn("--force_ocr", args)
            self.assertIn("--output_format", args)

    def test_marker_cli_contract_accepts_page_range(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "marker"

            def fake_run(args, **_kwargs):
                output.mkdir(parents=True, exist_ok=True)
                (output / "book.md").write_text("# Title", encoding="utf-8")
                return subprocess.CompletedProcess(args, 0, "", "")

            with patch("bookflow.adapters.run_command", side_effect=fake_run) as run:
                MarkerAdapter("marker_single").convert(root / "book.pdf", output, "off", "0-7")
            args = run.call_args.args[0]
            self.assertIn("--page_range", args)
            self.assertIn("0-7", args)

    def test_calibre_contract_is_input_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "book.epub"

            def fake_run(args, **_kwargs):
                destination.write_bytes(b"epub")
                return subprocess.CompletedProcess(args, 0, "", "")

            with patch("bookflow.adapters.run_command", side_effect=fake_run) as run:
                result = CalibreAdapter("ebook-convert").convert(root / "book.azw3", destination)
            self.assertEqual(result, destination)
            self.assertEqual(run.call_args.args[0], ["ebook-convert", str(root / "book.azw3"), str(destination)])

    def test_calibre_azw3_output_disables_duplicate_inline_toc(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "book.azw3"

            def fake_run(args, **_kwargs):
                destination.write_bytes(b"azw3")
                return subprocess.CompletedProcess(args, 0, "", "")

            with patch("bookflow.adapters.run_command", side_effect=fake_run) as run:
                result = CalibreAdapter("ebook-convert").convert(root / "book.epub", destination)
            self.assertEqual(result, destination)
            self.assertEqual(
                run.call_args.args[0],
                ["ebook-convert", str(root / "book.epub"), str(destination), "--no-inline-toc"],
            )

    def test_babelebook_adapter_keeps_provider_configurable(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "translated.epub"
            captured = {}

            def fake_run(args, **_kwargs):
                config_path = Path(args[args.index("--config") + 1])
                captured["config"] = json.loads(config_path.read_text(encoding="utf-8"))
                destination.write_bytes(b"epub")
                return subprocess.CompletedProcess(args, 0, "", "")

            config = Config(translation_model="configured-model")
            with patch("bookflow.adapters.run_command", side_effect=fake_run) as run:
                BabelEbookAdapter("babel-ebook").translate(root / "book.epub", destination, config)
            args = run.call_args.args[0]
            self.assertEqual(args[1], "translate")
            self.assertIn("--config", args)
            self.assertEqual(captured["config"]["provider"], "ollama")
            self.assertEqual(captured["config"]["model"], "configured-model")


if __name__ == "__main__":
    unittest.main()
