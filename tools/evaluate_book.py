#!/usr/bin/env python3
"""Run a resumable Qwen3.5 book-level consistency evaluation."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from bookflow.adapters import OllamaClient, OllamaTranslator
from bookflow.config import Config
from bookflow.epub import book_text, inspect_epub
from bookflow.memory import TranslationMemory, extract_terms, summarize_chapter


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def quality(segments: list[dict], terms: dict[str, str]) -> dict[str, object]:
    issues: list[str] = []
    warnings: list[str] = []
    for item in segments:
        original = item["original"]
        translated = item["translation"].strip()
        if not translated:
            issues.append(f"empty translation: {item['segment_id']}")
            continue
        if not re.fullmatch(r"\s*Chapter\s+\d+\s*", original, flags=re.IGNORECASE):
            numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", original))
            translated_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", translated))
            if numbers - translated_numbers:
                issues.append(f"possible number loss {item['segment_id']}: {sorted(numbers - translated_numbers)}")
        residual = sorted(set(re.findall(r"\b[A-Za-z]{4,}\b", translated)))
        if residual:
            issues.append(f"untranslated Latin text {item['segment_id']}: {residual}")
        for source, target in terms.items():
            if source.casefold() in original.casefold() and target not in translated:
                warnings.append(f"term mismatch {item['segment_id']}: {source} -> {target}")
    return {"ok": not issues, "issues": issues, "warnings": warnings}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("book", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:latest")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    book = inspect_epub(args.book)
    selected = book.segments[args.start : args.start + args.limit]
    if not selected:
        raise SystemExit("No segments selected")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"{slug(args.model)}-book.json"
    memory_path = args.output / "translation_memory.json"
    resume_result: dict | None = None
    if destination.exists() and not args.force:
        resume_result = json.loads(destination.read_text(encoding="utf-8"))
        if resume_result.get("status") == "completed" and len(resume_result.get("segments", [])) == len(selected):
            print(destination)
            return 0

    config = Config(
        ollama_url=args.ollama_url,
        translation_model=args.model,
        target_language="zh-CN",
        output_mode="translated",
        chunk_size=6000,
        review_enabled=False,
        glossary={},
        temperature=0.2,
        top_p=0.9,
        num_ctx=8192,
        think=False,
        prompt_version="book-eval-v2-consistency",
        consistency_enabled=True,
        term_extraction_enabled=True,
        chapter_summary_enabled=True,
        context_segments=2,
        context_chars=3000,
        term_extraction_chars=18000,
    )
    memory = TranslationMemory.load(
        memory_path,
        book_title=book.title,
        context_segments=config.context_segments,
        context_chars=config.context_chars,
    )
    if args.force and memory_path.exists():
        # A forced evaluation must rebuild the durable term registry as well
        # as the translations; otherwise an old, rejected term can survive.
        memory_path.unlink()
        memory = TranslationMemory.load(
            memory_path,
            book_title=book.title,
            context_segments=config.context_segments,
            context_chars=config.context_chars,
        )
    client = OllamaClient(args.ollama_url, args.model, options={"temperature": 0.2, "top_p": 0.9, "num_ctx": 8192}, think=config.think)
    if not memory.terms_extracted:
        memory.add_terms(extract_terms(
            client,
            title=book.title,
            source_text=book_text(book),
            source_language="en",
            target_language="zh-CN",
            max_chars=config.term_extraction_chars,
        ))
        memory.terms_extracted = True
        memory.save()

    translator = OllamaTranslator(client, config)
    result = resume_result or {
        "status": "running",
        "model": args.model,
        "source_book": str(args.book),
        "source_title": book.title,
        "selected_segments": len(selected),
        "selection": {"start": args.start, "limit": args.limit},
        "started_at": now(),
        "config": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_ctx": config.num_ctx,
            "think": config.think,
            "context_segments": config.context_segments,
            "context_chars": config.context_chars,
            "prompt_version": config.prompt_version,
        },
        "segments": [],
    }
    completed_before = len(result.get("segments", []))
    result["status"] = "running"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # Rebuild the in-memory rolling window from the checkpoint. The JSON
    # memory intentionally stores durable facts, while recent context is
    # reconstructed from the completed segment pairs.
    memory.recent = []
    for item in result.get("segments", []):
        memory.record(item["segment_id"], item["chapter"], item["translation"])
    start_all = time.perf_counter()
    for index, segment in enumerate(selected[completed_before:], start=completed_before + 1):
        if result["segments"] and result["segments"][-1]["chapter"] != segment.chapter:
            previous_chapter = result["segments"][-1]["chapter"]
            previous_items = [item for item in result["segments"] if item["chapter"] == previous_chapter]
            if previous_chapter not in memory.chapter_summaries:
                summary = summarize_chapter(
                    client,
                    title=book.title,
                    chapter=previous_chapter,
                    source_text="\n\n".join(item["original"] for item in previous_items),
                    translation_text="\n\n".join(item["translation"] for item in previous_items),
                )
                memory.set_summary(previous_chapter, summary)
                memory.save()
        started = time.perf_counter()
        context = memory.context_for(chapter=segment.chapter, source_text=segment.text)
        translation = translator.translate(segment, "en", context)
        item = {
            "index": index,
            "segment_id": segment.segment_id,
            "chapter": segment.chapter,
            "original": segment.text,
            "translation": translation,
            "context_fingerprint": memory.context_fingerprint(chapter=segment.chapter, source_text=segment.text),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        result["segments"].append(item)
        memory.record(segment.segment_id, segment.chapter, translation)
        memory.save()
        result["completed_segments"] = index
        result["memory_fingerprint"] = memory.fingerprint()
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    chapter_groups: dict[str, list[dict]] = {}
    for item in result["segments"]:
        chapter_groups.setdefault(item["chapter"], []).append(item)
    for chapter, items in chapter_groups.items():
        if chapter in memory.chapter_summaries:
            continue
        summary = summarize_chapter(
            client,
            title=book.title,
            chapter=chapter,
            source_text="\n\n".join(item["original"] for item in items),
            translation_text="\n\n".join(item["translation"] for item in items),
        )
        memory.set_summary(chapter, summary)
        memory.save()

    result["status"] = "completed"
    result["finished_at"] = now()
    result["elapsed_seconds"] = round(time.perf_counter() - start_all, 3)
    result["source_chars"] = sum(len(item["original"]) for item in result["segments"])
    result["translation_chars"] = sum(len(item["translation"]) for item in result["segments"])
    result["memory"] = {
        "terms": memory.terms,
        "chapter_summaries": memory.chapter_summaries,
        "fingerprint": memory.fingerprint(),
    }
    result["qa"] = quality(result["segments"], memory.terms)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = args.output / f"{slug(args.model)}-book.md"
    lines = [f"# {book.title} — book consistency evaluation", "", f"Model: `{args.model}`", ""]
    for item in result["segments"]:
        lines.extend([
            f"## {item['index']}. {item['segment_id']}", "",
            "### Original", "", item["original"], "",
            "### 简体中文", "", item["translation"], "",
        ])
    markdown.write_text("\n".join(lines), encoding="utf-8")
    report = args.output / "evaluation-report.md"
    qa = result["qa"]
    report.write_text(
        "\n".join([
            f"# {book.title} — Qwen3.5 long-form consistency evaluation", "",
            f"- Model: `{args.model}`", f"- Segments: {len(selected)}", f"- Source characters: {result['source_chars']}",
            f"- Translation characters: {result['translation_chars']}", f"- Time: {result['elapsed_seconds']} seconds",
            f"- Automatic QA: {'PASS' if qa['ok'] else '; '.join(qa['issues'])}",
            f"- Terminology warnings: {len(qa['warnings'])}", f"- Extracted/locked terms: {len(memory.terms)}", "",
            "## Book memory", "", json.dumps(result["memory"], ensure_ascii=False, indent=2), "",
            "## Review note", "", "This run uses persistent terminology, recent translated context, and chapter-summary memory. Inspect the paired Markdown and JSON before enabling whole-book batch output.", "",
        ]),
        encoding="utf-8",
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
