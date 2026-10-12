from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class TranslationMemory:
    """Persistent book-level terminology and rolling translation context.

    The memory is deliberately model-agnostic. It stores only compact facts,
    chapter summaries, and recent translated segments; the source EPUB stays
    in the normalized book and is never copied wholesale into every prompt.
    """

    path: Path
    book_title: str
    terms: dict[str, str] = field(default_factory=dict)
    chapter_summaries: dict[str, str] = field(default_factory=dict)
    recent: list[dict[str, str]] = field(default_factory=list)
    context_segments: int = 2
    context_chars: int = 3000
    version: int = 1
    terms_extracted: bool = False
    locked_terms: set[str] = field(default_factory=set, repr=False)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        book_title: str,
        configured_terms: dict[str, str] | None = None,
        context_segments: int = 2,
        context_chars: int = 3000,
    ) -> "TranslationMemory":
        configured_terms = configured_terms or {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = {}
            memory = cls(
                path=path,
                book_title=str(raw.get("book_title", book_title)),
                terms={str(k): str(v) for k, v in (raw.get("terms", {}) or {}).items()},
                chapter_summaries={str(k): str(v) for k, v in (raw.get("chapter_summaries", {}) or {}).items()},
                context_segments=int(raw.get("context_segments", context_segments)),
                context_chars=int(raw.get("context_chars", context_chars)),
                version=int(raw.get("version", 1)),
                terms_extracted=bool(raw.get("terms_extracted", False)),
            )
        else:
            memory = cls(path, book_title, context_segments=context_segments, context_chars=context_chars)
        # User configuration is authoritative over model-extracted entries.
        memory.terms = {**memory.terms, **configured_terms}
        memory.locked_terms = set(configured_terms)
        return memory

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "book_title": self.book_title,
            "terms": dict(sorted(self.terms.items(), key=lambda item: item[0].lower())),
            "chapter_summaries": self.chapter_summaries,
            "context_segments": self.context_segments,
            "context_chars": self.context_chars,
            "terms_extracted": self.terms_extracted,
            "fingerprint": self.fingerprint(),
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def fingerprint(self) -> str:
        return _digest({"terms": self.terms, "summaries": self.chapter_summaries, "version": self.version})

    def relevant_terms(self, source_text: str, *, limit: int = 24) -> dict[str, str]:
        lowered = source_text.casefold()
        matches = [(source, target) for source, target in self.terms.items() if source.casefold() in lowered]
        if len(matches) < limit:
            # A small global tail keeps recurring names consistent even when
            # the current paragraph uses a pronoun or an inflected form.
            for source, target in self.terms.items():
                pair = (source, target)
                if pair not in matches:
                    matches.append(pair)
                if len(matches) >= limit:
                    break
        return dict(matches[:limit])

    def context_for(self, *, chapter: str, source_text: str) -> str:
        lines = [
            f"Book title: {self.book_title}",
            f"Current chapter: {chapter}",
            "Use this information only for consistency. Do not translate or repeat this context.",
        ]
        if self.chapter_summaries:
            summaries = list(self.chapter_summaries.items())[-4:]
            lines.append("Prior chapter summaries:")
            lines.extend(f"- {key}: {value}" for key, value in summaries)
        terms = self.relevant_terms(source_text)
        if terms:
            lines.append("Locked terminology:")
            lines.extend(f"- {source} => {target}" for source, target in terms.items())
        if self.recent:
            lines.append("Recent translated context:")
            recent_chars = 0
            for item in reversed(self.recent[-self.context_segments :]):
                text = item["translation"]
                if recent_chars + len(text) > self.context_chars:
                    text = text[: max(0, self.context_chars - recent_chars)]
                if not text:
                    break
                lines.append(f"- {item['segment_id']}: {text}")
                recent_chars += len(text)
                if recent_chars >= self.context_chars:
                    break
        return "\n".join(lines)

    def context_fingerprint(self, *, chapter: str, source_text: str) -> str:
        return _digest({"memory": self.fingerprint(), "context": self.context_for(chapter=chapter, source_text=source_text)})

    def record(self, segment_id: str, chapter: str, translation: str) -> None:
        self.recent.append({"segment_id": segment_id, "chapter": chapter, "translation": translation})
        # Do not carry a long previous chapter into the next chapter. The
        # persisted summary is the compact cross-chapter memory.
        if len(self.recent) > max(self.context_segments * 2, 4):
            self.recent = self.recent[-max(self.context_segments * 2, 4) :]

    def add_terms(self, terms: dict[str, str]) -> bool:
        changed = False
        for source, target in terms.items():
            source = source.strip()
            target = target.strip()
            if source in self.locked_terms:
                continue
            if source and target and self.terms.get(source) != target:
                self.terms[source] = target
                changed = True
        return changed

    def set_summary(self, chapter: str, summary: str) -> bool:
        summary = re.sub(r"\s+", " ", summary).strip()
        if not summary or self.chapter_summaries.get(chapter) == summary:
            return False
        self.chapter_summaries[chapter] = summary[:1200]
        return True


def parse_term_response(raw: str) -> dict[str, str]:
    """Parse a tolerant JSON term-extraction response."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    entries = value.get("terms", []) if isinstance(value, dict) else value
    result: dict[str, str] = {}
    if isinstance(entries, dict):
        entries = [{"source": key, "target": val} for key, val in entries.items()]
    if not isinstance(entries, list):
        return result
    for item in entries:
        if isinstance(item, dict):
            source = item.get("source") or item.get("term")
            target = item.get("target") or item.get("translation")
            if isinstance(source, str) and isinstance(target, str):
                result[source.strip()] = target.strip()
    return result


def extract_terms(client: Any, *, title: str, source_text: str, source_language: str, target_language: str, max_chars: int = 18000) -> dict[str, str]:
    prompt = (
        "Extract a compact locked terminology list for a literary book translation. "
        "Return JSON only with this shape: {\"terms\":[{\"source\":\"...\",\"target\":\"...\"}]}. "
        "Include recurring people, places, organizations, titles, and distinctive technical terms. "
        "Use the most natural Simplified Chinese translation. Do not include ordinary words. "
        f"Book: {title}\nSource language: {source_language}\nTarget language: {target_language}\n"
        f"Source excerpt:\n{source_text[:max_chars]}"
    )
    terms = parse_term_response(client.chat("", prompt, json_format=True))
    mixed = {source: target for source, target in terms.items() if re.search(r"[A-Za-z]{3,}", target)}
    if mixed:
        repair_prompt = (
            "Correct these book terminology translations. Return JSON only with the same shape. "
            "Every target must be fully translated into natural Simplified Chinese; do not leave English words in targets. "
            f"Book: {title}\nEntries:\n{json.dumps(mixed, ensure_ascii=False)}"
        )
        repaired = parse_term_response(client.chat("", repair_prompt, json_format=True))
        for source in mixed:
            target = repaired.get(source, "")
            if target and not re.search(r"[A-Za-z]{3,}", target):
                terms[source] = target
            else:
                # Never lock a mixed-language term into the book memory. The
                # model can still translate it in context, while a human can
                # add an explicit override in config.yaml later.
                terms.pop(source, None)
    return terms


def summarize_chapter(client: Any, *, title: str, chapter: str, source_text: str, translation_text: str) -> str:
    prompt = (
        "Write a concise Chinese continuity summary for a literary translation memory. "
        "Mention only stable facts: people, relationships, places, events, and narrative tone. "
        "Do not critique the translation and do not invent facts. Return plain Chinese in at most 180 words.\n"
        f"Book: {title}\nChapter: {chapter}\n"
        f"Original excerpt:\n{source_text[:5000]}\n\nTranslation excerpt:\n{translation_text[:5000]}"
    )
    return client.chat("", prompt)
