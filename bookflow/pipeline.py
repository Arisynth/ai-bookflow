from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .adapters import (
    BabelEbookAdapter,
    CalibreAdapter,
    MarkerAdapter,
    OllamaClient,
    OllamaReviewer,
    OllamaTranslator,
)
from .config import Config
from .epub import EpubBook, book_text, build_translated_epub, inspect_epub, markdown_to_epub, validate_epub
from .errors import BookFlowError
from .inspect import InputInfo, inspect_input
from .language import detect_language
from .memory import TranslationMemory, summarize_chapter, valid_term_exception
from .memory import _digest
from .preparation import prepare_book, write_json
from .book_review import adjacent_source, chapter_review
from .qa import deterministic_qa, numeric_advisories, numeric_evidence


@dataclass
class JobState:
    job_id: str
    input_path: str
    input_sha256: str
    kind: str
    workdir: str
    status: str = "created"
    stage: str = "inspect"
    source_language: str | None = None
    translated: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    error: str | None = None
    skipped_translation: bool = False
    external_output: str | None = None
    quality_status: str = "not_evaluated"
    quality_summary: dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    def write(self, path: Path) -> None:
        self.updated_at = time.time()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def read(cls, path: Path) -> "JobState":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class Pipeline:
    def __init__(self, config: Config):
        config.validate()
        self.config = config
        self.memory: TranslationMemory | None = None
        self.review_contexts: dict[str, str] = {}
        self.chapter_findings: dict[str, list[dict[str, Any]]] = {}
        self.forced_review_ids: set[str] = set()

    def _paths(self, workdir: Path) -> dict[str, Path]:
        paths = {name: workdir / name for name in ("source", "normalized", "metadata", "translations", "reviews", "output", "logs")}
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def _log(self, state: JobState, event: str, **details: Any) -> None:
        log_path = Path(state.workdir) / "logs" / "events.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"time": time.time(), "job_id": state.job_id, "stage": state.stage, "event": event, **details}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _new_state(self, info: InputInfo, workdir: Path) -> JobState:
        job_id = hashlib.sha256(f"{info.path}\0{info.sha256}".encode()).hexdigest()[:16]
        return JobState(job_id, str(info.path), info.sha256, info.kind, str(workdir))

    def start(self, input_path: Path, *, output_dir: Path | None = None) -> JobState:
        info = inspect_input(input_path)
        job_id = hashlib.sha256(f"{info.path}\0{info.sha256}".encode()).hexdigest()[:16]
        workdir = self.config.workspace / job_id
        state_path = workdir / "state.json"
        if state_path.exists():
            state = JobState.read(state_path)
            if state.input_sha256 != info.sha256:
                raise BookFlowError(f"任务 ID 已存在但输入文件内容不同: {job_id}")
        else:
            workdir.mkdir(parents=True, exist_ok=True)
            state = self._new_state(info, workdir)
            state.write(state_path)
        return self.run(state, output_dir=output_dir)

    def resume(self, job_id: str, *, output_dir: Path | None = None) -> JobState:
        state_path = self.config.workspace / job_id / "state.json"
        if not state_path.exists():
            raise BookFlowError(f"找不到任务: {job_id}")
        return self.run(JobState.read(state_path), output_dir=output_dir)

    def run(self, state: JobState, *, output_dir: Path | None = None) -> JobState:
        workdir = Path(state.workdir)
        paths = self._paths(workdir)
        state_path = workdir / "state.json"
        state.status = "running"
        state.error = None
        state.outputs = []
        state.skipped_translation = False
        state.external_output = None
        state.quality_status = "not_evaluated"
        state.quality_summary = {}
        self.memory = None
        self.review_contexts = {}
        self.chapter_findings = {}
        state.write(state_path)
        self._log(state, "started")
        try:
            normalized = self._normalize(state, paths, state_path)
            book = inspect_epub(normalized)
            detected = self.config.source_language or detect_language(book_text(book))
            state.source_language = detected
            (paths["metadata"] / "book.json").write_text(
                json.dumps({"title": book.title, "source_language": detected, "segments": len(book.segments)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            state.write(state_path)
            context = self._context(book)
            translations = self._translate(state, book, paths, state_path, context)
            self._review(state, book, translations, paths, state_path, context, finalize=False)
            if not state.skipped_translation and not state.external_output and self.config.output_mode != "original":
                self._finish_book(state, book, translations, paths, state_path, context)
            epub_output = self._build(state, normalized, translations, paths, state_path, output_dir)
            outputs = [epub_output]
            if "azw3" in self.config.output_formats:
                azw3 = Path(epub_output).parent / f"{Path(state.input_path).stem}.azw3"
                CalibreAdapter(self.config.calibre_command).convert(epub_output, azw3)
                outputs.append(azw3)
            state.outputs = [str(item) for item in outputs]
            state.stage = "validate"
            state.write(state_path)
            if self.config.validate_epub:
                errors = validate_epub(epub_output)
                if errors:
                    raise BookFlowError("EPUB 校验失败:\n" + "\n".join(errors))
                self._validate_artifact(state, epub_output, paths)
            if state.quality_status == "auto_accepted":
                state.quality_status = "finalized" if self.config.validate_epub else "draft"
            if self.config.strict_review and not state.skipped_translation and not state.external_output and state.quality_status != "finalized":
                raise BookFlowError("严格验收要求全部质量检查及 EPUBCheck 通过；请查看 reviews/ 下的报告")
            self._write_acceptance(state, translations, paths)
            state.status = "completed"
            state.stage = "done"
            state.write(state_path)
            self._log(state, "completed", outputs=state.outputs)
            return state
        except Exception as exc:
            state.status = "failed"
            state.error = str(exc)
            state.quality_status = "blocked"
            # Invalidate a previous acceptance even if failure happened before
            # rebuilding. Old files may still exist, but cannot be this run's final.
            if "translations" in locals():
                self._write_acceptance(state, translations, paths)
            else:
                write_json(paths["reviews"] / "acceptance.json", {"quality_status": "blocked", "error": str(exc), "outputs": []})
            state.write(state_path)
            self._log(state, "failed", error=str(exc))
            if isinstance(exc, BookFlowError):
                raise
            raise BookFlowError(str(exc)) from exc

    def _normalize(self, state: JobState, paths: dict[str, Path], state_path: Path) -> Path:
        destination = paths["normalized"] / "book.epub"
        if destination.exists():
            state.stage = "normalize"
            state.write(state_path)
            return destination
        source = paths["source"] / Path(state.input_path).name
        if not source.exists():
            shutil.copy2(state.input_path, source)
        if state.kind == "epub":
            shutil.copy2(source, destination)
        elif state.kind in {"mobi", "azw", "azw3"}:
            CalibreAdapter(self.config.calibre_command).convert(source, destination)
        elif state.kind == "pdf":
            marker_dir = paths["normalized"] / "marker"
            markdown = MarkerAdapter(self.config.marker_command).convert(
                source,
                marker_dir,
                self.config.ocr_mode,
                self.config.marker_page_range,
            )
            markdown_to_epub(markdown, destination, Path(state.input_path).stem)
        else:
            raise BookFlowError(f"不支持的输入类型: {state.kind}")
        state.stage = "normalize"
        state.write(state_path)
        self._log(state, "normalized", path=str(destination))
        return destination

    def _context(self, book: EpubBook) -> str:
        # Do not put raw source paragraphs into the context field: models may
        # translate the context as well as the target segment. The paragraph
        # itself is supplied separately by OllamaTranslator.
        return f"Title: {book.title}\nTranslate faithfully. Preserve the author's voice, genre and terminology."

    def _cache_key(self, text: str, source_language: str, context_fingerprint: str = "") -> str:
        payload = {
            "text": text,
            "source": source_language,
            "target": self.config.target_language,
            "model": self.config.translation_model,
            "prompt": self.config.prompt_version,
            "glossary": self.config.glossary,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "num_ctx": self.config.num_ctx,
            "think": self.config.think,
            "context": context_fingerprint,
            "consistency": self.config.consistency_enabled,
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def _translation_fingerprint(self, source_language: str, *, legacy: bool = False) -> str:
        payload = {
            "source": source_language,
            "target": self.config.target_language,
            "model": self.config.translation_model,
            "prompt": self.config.prompt_version,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "num_ctx": self.config.num_ctx,
            "think": self.config.think,
            "consistency": self.config.consistency_enabled,
            "context_segments": self.config.context_segments,
            "context_chars": self.config.context_chars,
        }
        if legacy:
            payload["glossary"] = self.config.glossary
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def _translate(self, state: JobState, book: EpubBook, paths: dict[str, Path], state_path: Path, context: str) -> dict[str, str]:
        state.stage = "translate"
        state.write(state_path)
        self._log(state, "translation_started", segments=len(book.segments))
        if self.config.output_mode == "original":
            state.skipped_translation = True
            return {}
        if state.source_language == "unknown":
            raise BookFlowError("无法自动识别原书语言，请使用 --source 指定语言")
        if state.source_language and state.source_language.lower().startswith("zh") and self.config.target_language.lower().startswith("zh"):
            state.skipped_translation = True
            translations = {segment.segment_id: segment.text for segment in book.segments}
            (paths["translations"] / "segments.json").write_text(json.dumps(translations, ensure_ascii=False, indent=2), encoding="utf-8")
            state.write(state_path)
            self._log(state, "translation_skipped", reason="target_is_source_language")
            return translations
        if self.config.translation_engine == "babelebook":
            if self.config.consistency_enabled or self.config.review_enabled or self.config.glossary or self.config.terminology or self.config.style:
                raise BookFlowError("BabelEbook 路径不支持统一术语、风格和全书验收；请使用 translation.engine: ollama，或明确关闭这些功能且移除规则配置")
            external = paths["translations"] / "babelebook.epub"
            BabelEbookAdapter(self.config.babelebook_command).translate(paths["normalized"] / "book.epub", external, self.config)
            state.external_output = str(external)
            state.write(state_path)
            return {}
        if not book.segments:
            raise BookFlowError("原书没有可翻译段落；请检查解析结果或 OCR")
        memory = TranslationMemory.load(
            paths["metadata"] / "translation_memory.json",
            book_title=book.title,
            configured_terms=self.config.glossary,
            context_segments=self.config.context_segments,
            context_chars=self.config.context_chars,
        )
        cache_path = paths["translations"] / "cache.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        translations_path = paths["translations"] / "segments.json"
        translations: dict[str, str] = json.loads(translations_path.read_text(encoding="utf-8")) if translations_path.exists() else {}
        old_translations = dict(translations)
        for segment in book.segments:
            old_id = book.legacy_ids.get(segment.segment_id)
            if segment.segment_id not in translations and old_id in old_translations:
                translations[segment.segment_id] = old_translations[old_id]
        segment_ids = {s.segment_id for s in book.segments}
        translations = {key: value for key, value in translations.items() if key in segment_ids}
        meta_path = paths["translations"] / "meta.json"
        fingerprint = self._translation_fingerprint(state.source_language or "unknown")
        previous_fingerprint = json.loads(meta_path.read_text(encoding="utf-8")).get("fingerprint") if meta_path.exists() else None
        legacy_fingerprint = self._translation_fingerprint(state.source_language or "unknown", legacy=True)
        if previous_fingerprint and previous_fingerprint not in {fingerprint, legacy_fingerprint}:
            translations = {}
            state.translated = []
            memory.chapter_summaries = {}
            memory.summary_hashes = {}
        meta_path.write_text(json.dumps({"fingerprint": fingerprint}, indent=2), encoding="utf-8")
        options = {"temperature": self.config.temperature}
        if self.config.top_p is not None:
            options["top_p"] = self.config.top_p
        if self.config.num_ctx is not None:
            options["num_ctx"] = self.config.num_ctx
        client = OllamaClient(
            self.config.ollama_url,
            self.config.translation_model,
            options=options,
            think=self.config.think,
            max_retries=self.config.request_retries,
            retry_backoff_seconds=self.config.retry_backoff_seconds,
        )
        translator = OllamaTranslator(client, self.config)
        if self.config.consistency_enabled:
            state.stage = "prepare"
            state.write(state_path)
            prepare_book(book, memory, client, self.config, paths["metadata"] / "preparation.json")
            self.memory = memory
            state.stage = "translate"
            state.write(state_path)
        elif self.config.style or self.config.terminology or self.config.glossary:
            memory.style = self.config.style
            memory.style_origins = {k: "human" for k in memory.style}
            memory.entries = {**{k: {**v, "origin": "human"} for k, v in self.config.terminology.items()},
                              **{k: {"target": v, "origin": "human"} for k, v in self.config.glossary.items()}}
            memory.terms = {k: v["target"] for k, v in memory.entries.items()}
            self.memory = memory
        policy_path = paths["translations"] / "policy.json"
        old_policies = json.loads(policy_path.read_text()) if policy_path.exists() else {}
        policies = {}
        # Existing translations are preserved during policy upgrades, then
        # re-reviewed locally. Turning review off requires affected retranslation.
        for segment in book.segments:
            policy = memory.policy_fingerprint(segment.text) if self.memory else ""
            policies[segment.segment_id] = policy
            if old_policies.get(segment.segment_id) != policy:
                self.forced_review_ids.add(segment.segment_id)
                if not self.config.review_enabled:
                    translations.pop(segment.segment_id, None)
        # Keep old revisions for unaffected segments when only human rules change.
        memory.recent = []
        self.review_contexts = {s.segment_id: self._segment_context(book, i, context)
                                for i, s in enumerate(book.segments)}
        for index, segment in enumerate(book.segments):
            if self.config.consistency_enabled and self.config.chapter_summary_enabled and memory.recent:
                previous_chapter = memory.recent[-1]["chapter"]
                if previous_chapter != segment.chapter and previous_chapter not in memory.chapter_summaries:
                    previous_segments = [item for item in book.segments if item.chapter == previous_chapter]
                    source_text = "\n\n".join(item.text for item in previous_segments)
                    translation_text = "\n\n".join(translations.get(item.segment_id, "") for item in previous_segments)
                    if translation_text:
                        try:
                            memory.set_summary(
                                previous_chapter,
                                memory.chapter_briefs.get(previous_chapter) or summarize_chapter(client, title=book.title, chapter=previous_chapter, source_text=source_text, translation_text=translation_text),
                            )
                            memory.save()
                        except Exception as exc:
                            self._log(state, "chapter_summary_failed", chapter=previous_chapter, error=str(exc))
            segment_context = self.review_contexts[segment.segment_id]
            if self.memory:
                prior_chapters = list(dict.fromkeys(s.chapter for s in book.segments[:index]
                                                   if s.chapter != segment.chapter))[-2:]
                notes = [f"{key}: {memory.chapter_summaries[key][:600]}" for key in prior_chapters
                         if key in memory.chapter_summaries]
                if notes:
                    segment_context += "\nPrior chapter continuity (source wins over summary):\n" + "\n".join(notes)
            if self.memory and memory.recent and self.config.context_segments:
                recent = "\n".join(item["translation"] for item in memory.recent[-self.config.context_segments:])
                if self.config.context_chars:
                    segment_context += "\nRecent translated context (not authoritative):\n" + recent[-self.config.context_chars:]
            context_fingerprint = _digest(segment_context) if self.memory else ""
            if segment.segment_id in translations:
                memory.record(segment.segment_id, segment.chapter, translations[segment.segment_id])
                continue
            key = self._cache_key(segment.text, state.source_language or "unknown", context_fingerprint)
            write_json(paths["translations"] / "contexts" / f"{segment.segment_id}.json", {
                "source": segment.text, "context": segment_context, "cache_key": key,
                "matched_terms": memory.relevant_terms(segment.text) if self.memory else {},
                "context_chars": len(segment_context),
            })
            if key in cache:
                translated = str(cache[key])
            else:
                translated = translator.translate(segment, state.source_language or "unknown", segment_context)
                if not translated:
                    raise BookFlowError(f"Ollama 返回空译文: {segment.segment_id}")
                cache[key] = translated
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            translations[segment.segment_id] = translated
            if self.config.consistency_enabled:
                memory.record(segment.segment_id, segment.chapter, translated)
                memory.save()
            state.translated = sorted(translations)
            translations_path.write_text(json.dumps(translations, ensure_ascii=False, indent=2), encoding="utf-8")
            state.write(state_path)
            self._log(state, "segment_translated", segment_id=segment.segment_id, completed=len(translations))
        if self.config.consistency_enabled and self.config.chapter_summary_enabled:
            chapters: dict[str, list[Any]] = {}
            for segment in book.segments:
                chapters.setdefault(segment.chapter, []).append(segment)
            for chapter, segments in chapters.items():
                if chapter in memory.chapter_summaries:
                    continue
                source_text = "\n\n".join(segment.text for segment in segments)
                translation_text = "\n\n".join(translations.get(segment.segment_id, "") for segment in segments)
                try:
                    summary = memory.chapter_briefs.get(chapter) or summarize_chapter(client, title=book.title, chapter=chapter, source_text=source_text, translation_text=translation_text)
                    if memory.set_summary(chapter, summary):
                        memory.save()
                except Exception as exc:
                    self._log(state, "chapter_summary_failed", chapter=chapter, error=str(exc))
        write_json(policy_path, policies)
        write_json(translations_path, translations)
        state.translated = sorted(translations)
        return translations

    def _review(self, state: JobState, book: EpubBook, translations: dict[str, str], paths: dict[str, Path], state_path: Path, context: str, *, finalize: bool = True) -> None:
        state.stage = "review"
        state.write(state_path)
        self._log(state, "review_started")
        if state.external_output or state.skipped_translation:
            return
        checkpoint_path = paths["reviews"] / "review_checkpoint.json"
        audit_dir = paths["reviews"] / "segments"
        audit_dir.mkdir(parents=True, exist_ok=True)
        review_version = "book-policy-v3"

        def write_json(path: Path, value: Any) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)

        def text_hash(text: str) -> str:
            return hashlib.sha256(text.encode()).hexdigest()

        def review_context(segment_id: str) -> str:
            value = self.review_contexts.get(segment_id, context)
            if segment_id in self.chapter_findings:
                value += "\nChapter-level allegations to verify against CURRENT text and adjacent source:\n" + json.dumps(self.chapter_findings[segment_id], ensure_ascii=False)
            return value

        def checked_review(segment: Any, candidate: str, reviewer: Any) -> dict[str, Any]:
            current_context = review_context(segment.segment_id)
            missing_terms = self.memory.term_violations(segment.text, candidate) if self.memory else {}
            if missing_terms:
                current_context += "\nMissing required terms; fix unless a source-supported automatic-term exception applies:\n" + json.dumps(missing_terms, ensure_ascii=False)
            hints = numeric_advisories(segment.text, candidate)
            if hints:
                current_context += "\nNumeric verification:\n" + "\n".join(hints)
            facts = numeric_evidence(segment.text, candidate)
            if facts["source"] or facts["translation"]:
                current_context += "\nComputed numeric magnitudes and exact currency evidence (do not invent conversion errors; still check ownership, sign and percent):\n" + json.dumps(facts, ensure_ascii=False)
            result = reviewer.review(segment.text, candidate, current_context)
            # A model approval cannot override lost or duplicated footnote anchors.
            hard_issues = deterministic_qa([segment], {segment.segment_id: candidate}).issues
            if hard_issues:
                result = {**result, "ok": False, "severity": "major",
                          "issues": [*result.get("issues", []), *hard_issues],
                          "suggestion": result.get("suggestion", "") + " Fix the deterministic defects; preserve numbers and all footnote placeholders in source order."}
            exceptions = result.get("terminology_exceptions", {})
            remaining = {k: v for k, v in missing_terms.items() if not (
                valid_term_exception(exceptions.get(k), segment.text) and self.memory and self.memory.entries.get(k, {}).get("origin") != "human")}
            if remaining:
                result = {**result, "ok": False, "severity": "major",
                          "issues": [*result.get("issues", []), "Required terminology: " + json.dumps(remaining, ensure_ascii=False)],
                          "suggestion": result.get("suggestion", "") + " Apply the required book terminology without changing meaning."}
            return result
        review: dict[str, Any] = {"deterministic": {}, "llm": []}
        if checkpoint_path.exists():
            try:
                candidate = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if isinstance(candidate, dict) and isinstance(candidate.get("llm"), list):
                    review["llm"] = candidate["llm"]
            except (OSError, json.JSONDecodeError):
                self._log(state, "review_checkpoint_ignored")
        def completed_review(item: Any) -> bool:
            if not isinstance(item, dict) or not item.get("segment_id"):
                return False
            expected_policy = _digest([self.review_contexts.get(item["segment_id"], context), self.config.review_model or self.config.translation_model,
                                      self.config.review_num_ctx, self.config.review_num_predict, self.config.review_think, self.config.dispute_enabled, self.config.dispute_think, review_version])
            if item.get("policy_sha256") != expected_policy:
                return False
            if item["segment_id"] in self.chapter_findings and item.get("chapter_findings_sha256") != _digest(self.chapter_findings[item["segment_id"]]):
                return False
            # Protocol failures must be retried on resume. This also handles
            # checkpoints written by older versions that incorrectly stored
            # malformed JSON as ok=true.
            if item.get("review_error") == "invalid_json":
                return False
            if "审校模型未返回合法 JSON" in [str(issue) for issue in item.get("issues", [])]:
                return False
            if not (
                isinstance(item.get("ok"), bool)
                and isinstance(item.get("issues"), list)
                and isinstance(item.get("suggestion"), str)
            ):
                return False
            if item["ok"] and (item["issues"] or item["suggestion"].strip()):
                return False
            if not item["ok"] and not item["issues"]:
                return False
            if item.get("review_version") != review_version and (
                item.get("revision_applied") or item.get("review_round", 0) > 0
            ):
                # Old whole-paragraph requests were fed through a chunking
                # translator. Recheck those outputs, even if marked accepted.
                return False
            # A failed review is never a completed review. It may have an
            # unresolved issue or may be an old checkpoint written before the
            # post-revision review loop existed. Re-review it on resume.
            if item.get("review_version") == review_version:
                if item.get("translation_sha256") != text_hash(translations.get(item["segment_id"], "")):
                    return False
                if (not self.config.strict_review and item.get("review_exhausted")
                        and item.get("selected_review") and not item.get("selection_error")):
                    return True
            return item["ok"]

        # The latest record wins. This prevents an older successful record
        # from masking a later protocol failure for the same segment.
        latest_by_id: dict[str, dict[str, Any]] = {}
        for item in review["llm"]:
            if isinstance(item, dict) and item.get("segment_id"):
                latest_by_id[str(item["segment_id"])] = item
        reviewed_ids = {segment_id for segment_id, item in latest_by_id.items() if completed_review(item)}
        if self.config.review_enabled and (self.config.review_model or self.config.translation_model):
            reviewer_options: dict[str, Any] = {"temperature": 0.0}
            if self.config.review_num_ctx is not None:
                reviewer_options["num_ctx"] = self.config.review_num_ctx
            if self.config.review_num_predict is not None:
                reviewer_options["num_predict"] = self.config.review_num_predict
            reviewer = OllamaReviewer(
                OllamaClient(
                    self.config.ollama_url,
                    self.config.review_model or self.config.translation_model,
                    options=reviewer_options,
                    think=self.config.review_think,
                    max_retries=self.config.request_retries,
                    retry_backoff_seconds=self.config.retry_backoff_seconds,
                ),
                max_parse_retries=self.config.request_retries,
                verifier=OllamaReviewer(self._editor_client(dispute=True), max_parse_retries=1) if self.config.dispute_enabled else None,
            )
            review_options: dict[str, Any] = {"temperature": self.config.temperature}
            if self.config.top_p is not None:
                review_options["top_p"] = self.config.top_p
            if self.config.review_num_ctx is not None or self.config.num_ctx is not None:
                review_options["num_ctx"] = self.config.review_num_ctx or self.config.num_ctx
            translator = OllamaTranslator(
                OllamaClient(
                    self.config.ollama_url,
                    self.config.translation_model,
                    options=review_options,
                    think=self.config.think,
                    max_retries=self.config.request_retries,
                    retry_backoff_seconds=self.config.retry_backoff_seconds,
                ),
                self.config,
            )
            severity_rank = {"none": 0, "minor": 1, "major": 2, "critical": 3}

            def round_limit(severity: str) -> int:
                if severity_rank.get(severity, 2) >= severity_rank["major"]:
                    return self.config.serious_revision_rounds
                return self.config.minor_revision_rounds

            def feedback_text(history: list[dict[str, Any]]) -> str:
                return "\n".join(
                    f"Round {entry['round']} | severity={entry.get('severity', 'major')} | "
                    f"issues: {'；'.join(entry.get('issues', []))} | "
                    f"suggestion: {entry.get('suggestion', '')}"
                    + (f" | revision_error: {entry['revision_error']}" if entry.get("revision_error") else "")
                    for entry in history
                )

            for segment in book.segments:
                if segment.segment_id in reviewed_ids:
                    continue
                stable_translation = translations[segment.segment_id]
                segment_context = review_context(segment.segment_id)
                policy_hash = _digest([self.review_contexts.get(segment.segment_id, context), self.config.review_model or self.config.translation_model,
                                       self.config.review_num_ctx, self.config.review_num_predict, self.config.review_think, self.config.dispute_enabled, self.config.dispute_think, review_version])
                candidate = stable_translation
                candidates = [stable_translation]
                history: list[dict[str, Any]] = []
                final_item: dict[str, Any] | None = None
                audit_path = audit_dir / f"{segment.segment_id}.json"
                # Checkpoint every model step. An interrupted segment retains
                # its candidates and original round budget on resume.
                start_round = 0
                if audit_path.exists():
                    saved = json.loads(audit_path.read_text(encoding="utf-8"))
                    if saved.get("complete"):
                        archive = audit_dir / "history" / f"{segment.segment_id}-{_digest(saved)}.json"
                        write_json(archive, saved)
                    if (saved.get("review_version") == review_version
                            and saved.get("policy_sha256") == policy_hash
                            and saved.get("chapter_findings_sha256") == _digest(self.chapter_findings.get(segment.segment_id, []))
                            and not saved.get("complete") and saved.get("source") == segment.text
                            and saved.get("initial_translation") == stable_translation):
                        candidates = saved["candidates"]
                        candidate = saved["current_translation"]
                        history = saved["history"]
                        start_round = saved["next_round"]
                        history = [entry for entry in history if entry["round"] < start_round]

                def save_audit(next_round: int, complete: bool = False) -> None:
                    write_json(audit_path, {
                        "review_version": review_version, "source": segment.text,
                        "policy_sha256": policy_hash,
                        "chapter_findings_sha256": _digest(self.chapter_findings.get(segment.segment_id, [])),
                        "initial_translation": stable_translation, "candidates": candidates,
                        "context": segment_context,
                        "current_translation": candidate, "history": history,
                        "next_round": next_round, "complete": complete,
                        "persisted_translation": translations[segment.segment_id] if complete else None,
                        "result": final_item,
                    })

                # A revision is provisional until a fresh review accepts it.
                # Minor issues get two revision rounds; major/critical issues
                # can escalate to the configured serious limit. Every round's
                # feedback is carried into the next translator prompt.
                for revision_round in range(start_round, self.config.serious_revision_rounds + 1):
                    # Fresh source-vs-candidate review prevents stale feedback
                    # from being copied as if it still described this version.
                    result = checked_review(segment, candidate, reviewer)
                    severity = str(result.get("severity", "major" if not result.get("ok") else "none"))
                    current_feedback = {
                        "round": revision_round,
                        "candidate_index": candidates.index(candidate),
                        "ok": bool(result.get("ok")),
                        "severity": severity,
                        "issues": [str(issue) for issue in result.get("issues", [])],
                        "suggestion": str(result.get("suggestion", "")),
                    }
                    if result.get("review_error"):
                        current_feedback["review_error"] = result["review_error"]
                    if result.get("preliminary_review"):
                        current_feedback["preliminary_review"] = result["preliminary_review"]
                    history.append(current_feedback)
                    save_audit(revision_round)
                    item = {
                        "segment_id": segment.segment_id,
                        **result,
                        "review_round": revision_round,
                        "severity": severity,
                        "review_history": history,
                    }
                    if result.get("review_error") == "invalid_json":
                        final_item = item
                        break
                    if result.get("ok") is True:
                        if candidate != stable_translation:
                            translations[segment.segment_id] = candidate
                            write_json(paths["translations"] / "segments.json", translations)
                            item["revision_applied"] = True
                        final_item = item
                        break

                    suggestion = str(result.get("suggestion", "")).strip()
                    # Once serious defects are fixed, minor style issues must
                    # not inherit a five-round budget from an obsolete verdict.
                    limit = round_limit(severity)
                    # Two consecutive unchanged/cycling edits provide no new
                    # candidate. Select and recheck the best rather than spend
                    # the remaining budget oscillating between identical texts.
                    stalled_edits = history[:-1][-2:]
                    if len(stalled_edits) == 2 and all(entry.get("revision_error") in {
                        "translator_returned_no_change", "translator_returned_previous_candidate"
                    } for entry in stalled_edits):
                        limit = min(limit, revision_round)
                    if revision_round >= limit or not suggestion:
                        # The final judge sees every candidate and the complete
                        # feedback history. This lets minor issues pass with
                        # the least-bad candidate and gives serious issues a
                        # principled best-candidate choice after five rounds.
                        selection = reviewer.select_best(
                            segment.text,
                            segment_context,
                            candidates,
                            feedback_text(history),
                        )
                        selected_index = int(selection.get("selected_index", 0))
                        if not 0 <= selected_index < len(candidates):
                            selected_index = 0
                        selected = candidates[selected_index]
                        # Selection reasons are not verdicts. Independently
                        # review the exact text that will be persisted.
                        selected_review = checked_review(segment, selected, reviewer)
                        item.update(selected_review)
                        item["selected_review"] = selected_review
                        item["selection"] = selection
                        if selected != stable_translation and (selected_review.get("ok") or not self.config.strict_review):
                            translations[segment.segment_id] = selected
                            write_json(paths["translations"] / "segments.json", translations)
                            item["revision_applied"] = True
                        else:
                            item["revision_applied"] = False
                        item["review_exhausted"] = not bool(selected_review.get("ok"))
                        item["selected_candidate_index"] = selected_index
                        item["candidate_count"] = len(candidates)
                        item["selection_reason"] = selection.get("reason", "")
                        if selection.get("selection_error"):
                            item["selection_error"] = selection["selection_error"]
                        final_item = item
                        break

                    stalled = bool(history[:-1]) and (
                        history[-2].get("revision_error") is not None
                        or history[-2].get("issues") == current_feedback["issues"])
                    revised = translator.revise(
                        segment, state.source_language or "unknown", segment_context,
                        stable_translation, candidate, history, rewrite=stalled,
                    )
                    if not revised or revised == candidate:
                        current_feedback["revision_error"] = "translator_returned_no_change"
                        if revision_round >= limit:
                            continue
                    else:
                        if revised in candidates:
                            current_feedback["revision_error"] = "translator_returned_previous_candidate"
                        candidate = revised
                        if candidate not in candidates:
                            candidates.append(candidate)
                    save_audit(revision_round + 1)

                if final_item is None:
                    # Defensive guard; the loop always sets this, but a
                    # checkpoint should never be left without a segment result.
                    final_item = {
                        "segment_id": segment.segment_id,
                        "ok": False,
                        "issues": ["审校闭环未产生最终结果"],
                        "suggestion": "",
                        "review_exhausted": True,
                    }

                # Replace stale/duplicate records so a recovered checkpoint
                # contains one authoritative result per segment.
                final_item["review_version"] = review_version
                final_item["policy_sha256"] = policy_hash
                final_item["chapter_findings_sha256"] = _digest(self.chapter_findings.get(segment.segment_id, []))
                final_item["translation_sha256"] = text_hash(translations[segment.segment_id])
                final_item["audit_file"] = str(audit_path)
                save_audit(revision_round, complete=not bool(final_item.get("review_error") or final_item.get("selection_error")))
                review["llm"] = [
                    existing
                    for existing in review["llm"]
                    if not (isinstance(existing, dict) and existing.get("segment_id") == segment.segment_id)
                ]
                review["llm"].append(final_item)
                if completed_review(final_item):
                    reviewed_ids.add(segment.segment_id)
                write_json(checkpoint_path, {"llm": review["llm"]})
                self.forced_review_ids.discard(segment.segment_id)
        report = deterministic_qa(book.segments, translations)
        review["deterministic"] = report.as_dict()
        (paths["reviews"] / "quality.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
        if finalize and not report.ok:
            raise BookFlowError("确定性质量检查发现问题，详见 reviews/quality.json: " + "; ".join(report.issues[:5]))
        unresolved = [
            item
            for item in review["llm"]
            if isinstance(item, dict) and (item.get("review_exhausted") or item.get("review_error") == "invalid_json")
        ]
        if finalize and self.config.strict_review and unresolved:
            raise BookFlowError(
                f"LLM 审校仍有 {len(unresolved)} 个段落未通过，详见 reviews/quality.json；"
                "恢复任务后会重新审校这些段落"
            )
        # Keep the final checkpoint and candidate audit trail for inspection.

    def _segment_context(self, book: EpubBook, index: int, fallback: str) -> str:
        segment = book.segments[index]
        value = (f"Book: {book.title}\nChapter: {segment.chapter}\n" + self.memory.policy_for(segment.text)
                 if self.memory else fallback)
        if self.memory:
            value += "\nWhole-book overview (background only; exact source wins):\n" + self.memory.book_brief[:800]
            value += "\nCurrent chapter overview (background only):\n" + self.memory.chapter_briefs.get(segment.chapter, "")[:1000]
        return value + "\n" + adjacent_source(book, index)

    def _editor_client(self, *, dispute: bool = False) -> OllamaClient:
        return OllamaClient(self.config.ollama_url, self.config.review_model or self.config.translation_model,
                            options={"temperature": 0.0, "num_ctx": self.config.review_num_ctx or 16384,
                                     "num_predict": max(2048, self.config.review_num_predict or 2048)},
                            think=self.config.dispute_think if dispute else self.config.review_think,
                            max_retries=min(1, self.config.request_retries) if dispute else self.config.request_retries,
                            retry_backoff_seconds=self.config.retry_backoff_seconds)

    def _refresh_summaries(self, book: EpubBook, translations: dict[str, str], client: Any) -> None:
        if not self.memory or not self.config.chapter_summary_enabled:
            return
        chapters: dict[str, list[Any]] = {}
        for segment in book.segments:
            chapters.setdefault(segment.chapter, []).append(segment)
        for chapter, segments in chapters.items():
            fingerprint = _digest([(s.text, translations[s.segment_id]) for s in segments])
            if self.memory.summary_hashes.get(chapter) == fingerprint:
                continue
            summary = self.memory.chapter_briefs.get(chapter) or summarize_chapter(client, title=book.title, chapter=chapter,
                source_text="\n\n".join(s.text for s in segments),
                translation_text="\n\n".join(translations[s.segment_id] for s in segments))
            self.memory.set_summary(chapter, summary)
            self.memory.summary_hashes[chapter] = fingerprint
            self.memory.save()

    def _finish_book(self, state: JobState, book: EpubBook, translations: dict[str, str],
                     paths: dict[str, Path], state_path: Path, context: str) -> None:
        state.stage = "book_review"
        state.write(state_path)
        findings: list[dict[str, Any]] = []
        client = self._editor_client()
        if self.config.review_enabled and self.config.chapter_review_enabled:
            memory = self.memory or TranslationMemory(paths["metadata"] / "translation_memory.json", book.title)
            pass_path = paths["reviews"] / "book_pass.json"
            policy = _digest([[(s.segment_id, s.text) for s in book.segments], memory.terms, memory.entries,
                              memory.style, self.config.review_model or self.config.translation_model,
                              self.config.book_revision_rounds, self.config.chapter_review_chars,
                              self.config.review_think, self.config.dispute_enabled, self.config.dispute_think,
                              self.config.review_num_ctx, self.config.review_num_predict, "book-pass-v1"])
            saved = json.loads(pass_path.read_text()) if pass_path.exists() else {}
            if saved.get("policy") != policy or (saved.get("phase") == "complete" and saved.get("text_hash") != _digest(translations)):
                saved = {"policy": policy, "round": 0, "phase": "checking"}
            while True:
                self._refresh_summaries(book, translations, client)
                if saved["phase"] == "revising":
                    findings = saved["issues"]
                else:
                    findings = chapter_review(book, translations, memory, client,
                        paths["reviews"] / "chapter_checkpoint.json", chars=self.config.chapter_review_chars,
                        retries=self.config.request_retries,
                        verifier_client=self._editor_client(dispute=True) if self.config.dispute_enabled else None)
                if not findings or saved["round"] >= self.config.book_revision_rounds:
                    saved.update(phase="complete", text_hash=_digest(translations), issues=findings)
                    write_json(pass_path, saved)
                    break
                saved.update(phase="revising", issues=findings)
                write_json(pass_path, saved)
                self.chapter_findings = {}
                for finding in findings:
                    self.chapter_findings.setdefault(finding["segment_id"], []).append(finding)
                self._review(state, book, translations, paths, state_path, context, finalize=False)
                saved.update(round=saved["round"] + 1, phase="checking", text_hash=_digest(translations))
                write_json(pass_path, saved)
        else:
            self._refresh_summaries(book, translations, client)
        quality_path = paths["reviews"] / "quality.json"
        quality = json.loads(quality_path.read_text()) if quality_path.exists() else {"llm": []}
        report = deterministic_qa(book.segments, translations)
        ids = {s.segment_id for s in book.segments}
        latest = {r["segment_id"]: r for r in quality.get("llm", []) if r.get("segment_id") in ids}
        unresolved = [r for r in latest.values() if not r.get("ok") or r.get("selection_error") or r.get("review_error")
                      or r.get("translation_sha256") != hashlib.sha256(translations[r["segment_id"]].encode()).hexdigest()]
        term_findings = []
        if self.memory:
            for segment in book.segments:
                exceptions = latest.get(segment.segment_id, {}).get("terminology_exceptions", {})
                for term, target in self.memory.term_violations(segment.text, translations[segment.segment_id]).items():
                    if valid_term_exception(exceptions.get(term), segment.text) and self.memory.entries.get(term, {}).get("origin") != "human":
                        continue
                    term_findings.append({"segment_id": segment.segment_id, "term": term, "target": target})
        coverage = len(latest) == len(ids) and self.config.review_enabled
        complete_checks = coverage and self.config.chapter_review_enabled and self.config.consistency_enabled
        state.quality_status = ("blocked" if not report.ok else "auto_accepted"
            if complete_checks and not unresolved and not findings and not term_findings else "draft")
        state.quality_summary = {"segments": len(ids), "reviewed": len(latest), "unresolved_segments": len(unresolved),
            "chapter_issues": len(findings), "terminology_issues": len(term_findings), "hard_errors": len(report.issues),
            "review_enabled": self.config.review_enabled, "chapter_review_enabled": self.config.chapter_review_enabled,
            "preparation_enabled": self.config.consistency_enabled}
        quality.update(deterministic=report.as_dict(), chapter=findings, terminology=term_findings,
                       quality_status=state.quality_status, summary=state.quality_summary)
        write_json(quality_path, quality)
        self._write_acceptance(state, translations, paths)
        state.write(state_path)
        if not report.ok:
            raise BookFlowError("确定性质量检查仍有错误，已阻止输出：" + "; ".join(report.issues[:5]))
        if self.config.strict_review and state.quality_status != "auto_accepted":
            raise BookFlowError("全书尚未通过质量验收；诊断、候选和恢复检查点已保存")

    def _validate_artifact(self, state: JobState, output: Path, paths: dict[str, Path]) -> None:
        from .adapters import run_command
        command = self.config.epubcheck_command
        result: dict[str, Any] = {"internal": "passed", "epubcheck": "not_installed"}
        if command and shutil.which(command):
            try:
                checked = run_command([command, str(output)])
                result.update(epubcheck="passed", details=(checked.stdout + checked.stderr)[-8000:])
            except BookFlowError as exc:
                result.update(epubcheck="failed", details=str(exc))
                state.quality_status = "blocked"
                write_json(paths["reviews"] / "artifact_validation.json", result)
                raise
        state.quality_summary["epubcheck"] = result["epubcheck"]
        # Structural checks alone do not certify typography on a physical reader.
        result["visual_inspection"] = "not_performed"
        write_json(paths["reviews"] / "artifact_validation.json", result)
        if result["epubcheck"] != "passed" and state.quality_status == "auto_accepted":
            state.quality_status = "draft"

    def _write_acceptance(self, state: JobState, translations: dict[str, str], paths: dict[str, Path]) -> None:
        data = {"quality_status": state.quality_status, "summary": state.quality_summary,
                "translation_sha256": hashlib.sha256(json.dumps(translations, sort_keys=True, ensure_ascii=False).encode()).hexdigest(), "model": self.config.translation_model,
                "review_model": self.config.review_model or self.config.translation_model,
                "policy": {"terms": self.memory.entries, "style": self.memory.style} if self.memory else {},
                "outputs": [{"path": str(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                            for p in state.outputs if Path(p).exists()],
                "human_approval": False, "acceptance_method": "local_model_and_automated_checks"}
        write_json(paths["reviews"] / "acceptance.json", data)
        quality_path = paths["reviews"] / "quality.json"
        if quality_path.exists():
            quality = json.loads(quality_path.read_text())
            quality.update(quality_status=state.quality_status, summary=state.quality_summary)
            write_json(quality_path, quality)
        for output in state.outputs:
            write_json(Path(output).with_suffix(Path(output).suffix + ".quality.json"), data)

    def _build(self, state: JobState, normalized: Path, translations: dict[str, str], paths: dict[str, Path], state_path: Path, output_dir: Path | None) -> Path:
        state.stage = "build"
        state.write(state_path)
        self._log(state, "build_started")
        target_dir = (output_dir or paths["output"]).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        destination = target_dir / f"{Path(state.input_path).stem}.epub"
        if state.external_output:
            shutil.copy2(state.external_output, destination)
        elif state.skipped_translation or self.config.output_mode == "original":
            shutil.copy2(normalized, destination)
        else:
            build_translated_epub(normalized, destination, translations, self.config.output_mode)
        return destination


def load_state(config: Config, job_id: str) -> JobState:
    path = config.workspace / job_id / "state.json"
    if not path.exists():
        raise BookFlowError(f"找不到任务: {job_id}")
    return JobState.read(path)
