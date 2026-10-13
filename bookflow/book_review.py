"""Contextual chapter checks and a resumable, bounded whole-book pass."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .epub import EpubBook
from .memory import TranslationMemory, _digest
from .preparation import structured, write_json
from .qa import numeric_evidence
from .errors import BookFlowError


def adjacent_source(book: EpubBook, index: int, chars: int = 1600) -> str:
    previous = book.segments[index - 1].text[-chars:] if index else ""
    following = book.segments[index + 1].text[:chars] if index + 1 < len(book.segments) else ""
    return f"Previous source paragraph (context only):\n{previous}\nNext source paragraph (context only):\n{following}"


def chapter_review(book: EpubBook, translations: dict[str, str], memory: TranslationMemory,
                   client: Any, checkpoint: Path, *, chars: int, retries: int, verifier_client: Any = None) -> list[dict[str, Any]]:
    saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    windows: list[list[int]] = []
    current: list[int] = []
    size = 0
    for index, segment in enumerate(book.segments):
        cost = len(segment.text) + len(translations[segment.segment_id])
        if current and (size + cost > chars or book.segments[current[-1]].chapter != segment.chapter):
            windows.append(current)
            current, size = [], 0
        current.append(index)
        size += cost
    if current:
        windows.append(current)
    findings: list[dict[str, Any]] = []
    active_keys = []
    for indices in windows:
        records = [{"segment_id": book.segments[i].segment_id, "original": book.segments[i].text,
                    "translation": translations[book.segments[i].segment_id],
                    "computed_numeric_evidence": numeric_evidence(book.segments[i].text, translations[book.segments[i].segment_id])} for i in indices]
        sources = "\n".join(r["original"] for r in records)
        context = memory.policy_for(sources) + "\n" + adjacent_source(book, indices[0], 1000)
        context += "\n" + adjacent_source(book, indices[-1], 1000)
        chapter = book.segments[indices[0]].chapter
        context += "\nWhole-book overview (original evidence wins):\n" + memory.book_brief[:800]
        context += "\nCurrent chapter overview:\n" + memory.chapter_briefs.get(chapter, "")[:1000]
        context += "\nChapter continuity summary (fallible, original evidence wins):\n" + memory.chapter_summaries.get(chapter, "")
        key = _digest(["chapter-v1", records, context, getattr(client, "model", ""),
                       getattr(client, "think", None), getattr(client, "options", {}),
                       bool(verifier_client), getattr(verifier_client, "think", None)])
        active_keys.append(key)
        if key not in saved:
            allowed = {r["segment_id"] for r in records}
            prompt = (
                "Review this book passage across paragraphs. Check discourse references, argument continuity, "
                "consistent terms and style, factual/number ownership, missing or repeated content, and the author's voice. "
                "Compare original evidence. Neighbor context and summaries must not be translated. "
                "Do not require literal translation or flag optional preferences. A locked term takes precedence over synonym preference. "
                "Only report substantiated defects and identify the exact paragraph to revise. "
                "Use the computed numeric magnitudes and currencies; equivalent unit conversions are correct, "
                "e.g. 1.3 billion dollars = 13 亿美元, 100,000 pounds = 10 万英镑. "
                "Human locked terms override conflicting automatic style examples. Never flag use of the exact human target as a terminology violation. "
                'Return {"issues":[{"segment_id":"ID from Records","severity":"minor|major|critical",'
                '"source_quote":"exact source evidence","translation_quote":"exact translation evidence or empty for omission",'
                '"issue":"specific defect","suggestion":"actionable correction"}]}. '
                "Return an empty issues list when no defects remain.\n" + context
                + "\nRecords:\n" + json.dumps(records, ensure_ascii=False)
            )
            def valid(value: Any) -> bool:
                if not isinstance(value, dict) or not isinstance(value.get("issues"), list):
                    return False
                for issue in value["issues"]:
                    if not isinstance(issue, dict) or issue.get("segment_id") not in allowed:
                        return False
                    if issue.get("severity") not in {"minor", "major", "critical"}:
                        return False
                    if not all(isinstance(issue.get(k), str) for k in ("source_quote", "translation_quote", "issue", "suggestion")):
                        return False
                    record = next(r for r in records if r["segment_id"] == issue["segment_id"])
                    if not issue["source_quote"] or issue["source_quote"] not in record["original"]:
                        return False
                    if issue["translation_quote"] and issue["translation_quote"] not in record["translation"]:
                        return False
                    if not issue["issue"].strip() or not issue["suggestion"].strip():
                        return False
                return True
            saved[key] = structured(client, prompt, valid, retries)
            if saved[key]["issues"] and verifier_client is not None:
                preliminary = saved[key]["issues"]
                try:
                    saved[key] = structured(verifier_client,
                        "Verify these preliminary allegations against the actual source and current translation. "
                        "They are NOT facts. Remove unsupported claims and optional preferences; "
                        "return only substantiated remaining issues. Explicit referents established in neighboring "
                        "original paragraphs are not unsupported additions.\n" + json.dumps(preliminary, ensure_ascii=False)
                        + "\n\n" + prompt, valid, 1)
                except BookFlowError as exc:
                    saved[key]["verification_error"] = str(exc)
                saved[key]["preliminary_issues"] = preliminary
            write_json(checkpoint, saved)
        findings.extend(saved[key]["issues"])
    # Preserve useful retry checkpoints, but report only windows matching the
    # current exact text. Old findings can never masquerade as current failures.
    write_json(checkpoint.with_name("chapter_result.json"), {"window_keys": active_keys, "issues": findings,
               "translation_hash": _digest(translations), "policy_hash": _digest([memory.terms, memory.style])})
    return findings
