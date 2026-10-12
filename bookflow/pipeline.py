from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from collections import Counter
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
from .memory import TranslationMemory, extract_terms, summarize_chapter
from .qa import deterministic_qa


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
            self._review(state, book, translations, paths, state_path, context)
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
            state.status = "completed"
            state.stage = "done"
            state.write(state_path)
            self._log(state, "completed", outputs=state.outputs)
            return state
        except Exception as exc:
            state.status = "failed"
            state.error = str(exc)
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
        return f"Title: {book.title}\nThis is a literary translation task. Preserve the chapter's narrative voice and terminology."

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

    def _translation_fingerprint(self, source_language: str) -> str:
        payload = {
            "source": source_language,
            "target": self.config.target_language,
            "model": self.config.translation_model,
            "prompt": self.config.prompt_version,
            "glossary": self.config.glossary,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "num_ctx": self.config.num_ctx,
            "think": self.config.think,
            "consistency": self.config.consistency_enabled,
            "context_segments": self.config.context_segments,
            "context_chars": self.config.context_chars,
        }
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
            external = paths["translations"] / "babelebook.epub"
            BabelEbookAdapter(self.config.babelebook_command).translate(paths["normalized"] / "book.epub", external, self.config)
            state.external_output = str(external)
            state.write(state_path)
            return {}
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
        meta_path = paths["translations"] / "meta.json"
        fingerprint = self._translation_fingerprint(state.source_language or "unknown")
        previous_fingerprint = json.loads(meta_path.read_text(encoding="utf-8")).get("fingerprint") if meta_path.exists() else None
        if previous_fingerprint and previous_fingerprint != fingerprint:
            translations = {}
            state.translated = []
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
        if self.config.consistency_enabled and self.config.term_extraction_enabled and not memory.terms_extracted:
            try:
                extracted = extract_terms(
                    client,
                    title=book.title,
                    source_text=book_text(book),
                    source_language=state.source_language or "unknown",
                    target_language=self.config.target_language,
                    max_chars=self.config.term_extraction_chars,
                )
                memory.add_terms(extracted)
            except Exception as exc:
                # Translation remains usable if optional term extraction fails;
                # the configured glossary and rolling context still apply.
                self._log(state, "term_extraction_failed", error=str(exc))
            memory.terms_extracted = True
            memory.save()
        for segment in book.segments:
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
                                summarize_chapter(client, title=book.title, chapter=previous_chapter, source_text=source_text, translation_text=translation_text),
                            )
                            memory.save()
                        except Exception as exc:
                            self._log(state, "chapter_summary_failed", chapter=previous_chapter, error=str(exc))
            segment_context = memory.context_for(chapter=segment.chapter, source_text=segment.text) if self.config.consistency_enabled else context
            context_fingerprint = memory.context_fingerprint(chapter=segment.chapter, source_text=segment.text) if self.config.consistency_enabled else ""
            if segment.segment_id in translations:
                memory.record(segment.segment_id, segment.chapter, translations[segment.segment_id])
                continue
            key = self._cache_key(segment.text, state.source_language or "unknown", context_fingerprint)
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
                    summary = summarize_chapter(client, title=book.title, chapter=chapter, source_text=source_text, translation_text=translation_text)
                    if memory.set_summary(chapter, summary):
                        memory.save()
                except Exception as exc:
                    self._log(state, "chapter_summary_failed", chapter=chapter, error=str(exc))
        return translations

    def _review(self, state: JobState, book: EpubBook, translations: dict[str, str], paths: dict[str, Path], state_path: Path, context: str) -> None:
        state.stage = "review"
        state.write(state_path)
        self._log(state, "review_started")
        if state.external_output or state.skipped_translation:
            return
        checkpoint_path = paths["reviews"] / "review_checkpoint.json"
        audit_dir = paths["reviews"] / "segments"
        audit_dir.mkdir(parents=True, exist_ok=True)
        review_version = "whole-segment-v2"

        def write_json(path: Path, value: Any) -> None:
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)

        def text_hash(text: str) -> str:
            return hashlib.sha256(text.encode()).hexdigest()

        def checked_review(segment: Any, candidate: str, reviewer: Any) -> dict[str, Any]:
            result = reviewer.review(segment.text, candidate, context)
            # A model approval cannot override lost or duplicated footnote anchors.
            tokens = lambda text: Counter(re.findall(r"\[\[BOOKFLOW_FOOTNOTE_\d+\]\]", text))
            if not candidate.strip() or tokens(segment.text) != tokens(candidate):
                result = {**result, "ok": False, "severity": "major",
                          "issues": [*result.get("issues", []), "Empty translation or mismatched footnote placeholders"],
                          "suggestion": result.get("suggestion", "") + " Restore every source footnote placeholder exactly once per source occurrence."}
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
        if self.config.review_enabled and self.config.review_model:
            reviewer_options: dict[str, Any] = {"temperature": 0.0}
            if self.config.review_num_ctx is not None:
                reviewer_options["num_ctx"] = self.config.review_num_ctx
            if self.config.review_num_predict is not None:
                reviewer_options["num_predict"] = self.config.review_num_predict
            reviewer = OllamaReviewer(
                OllamaClient(
                    self.config.ollama_url,
                    self.config.review_model,
                    options=reviewer_options,
                    think=self.config.think,
                    max_retries=self.config.request_retries,
                    retry_backoff_seconds=self.config.retry_backoff_seconds,
                ),
                max_parse_retries=self.config.request_retries,
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
                    if (saved.get("review_version") == review_version
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
                        "initial_translation": stable_translation, "candidates": candidates,
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
                    if revision_round >= limit or not suggestion:
                        # The final judge sees every candidate and the complete
                        # feedback history. This lets minor issues pass with
                        # the least-bad candidate and gives serious issues a
                        # principled best-candidate choice after five rounds.
                        selection = reviewer.select_best(
                            segment.text,
                            context,
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
                        segment, state.source_language or "unknown", context,
                        stable_translation, candidate, history, rewrite=stalled,
                    )
                    if not revised or revised == candidate:
                        current_feedback["revision_error"] = "translator_returned_no_change"
                        if revision_round >= limit:
                            continue
                    else:
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
        report = deterministic_qa(book.segments, translations)
        review["deterministic"] = report.as_dict()
        (paths["reviews"] / "quality.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
        if not report.ok:
            raise BookFlowError("确定性质量检查发现问题，详见 reviews/quality.json: " + "; ".join(report.issues[:5]))
        unresolved = [
            item
            for item in review["llm"]
            if isinstance(item, dict) and (item.get("review_exhausted") or item.get("review_error") == "invalid_json")
        ]
        if self.config.strict_review and unresolved:
            raise BookFlowError(
                f"LLM 审校仍有 {len(unresolved)} 个段落未通过，详见 reviews/quality.json；"
                "恢复任务后会重新审校这些段落"
            )
        # Keep the final checkpoint and candidate audit trail for inspection.

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
