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


def term_occurs(term: str, text: str) -> bool:
    # Match complete Latin words; avoid e.g. 'art' in 'party'. Permit regular
    # plurals and possessives, while aliases cover irregular/abbreviated names.
    pattern = re.escape(term)
    if re.search(r"[A-Za-z]", term):
        pattern = r"(?<!\w)" + pattern + r"(?:s|es|['’]s)?(?!\w)"
    return bool(re.search(pattern, text, re.IGNORECASE))


def valid_term_exception(value: Any, source: str) -> bool:
    return (isinstance(value, dict) and value.get("kind") in {"different_sense", "bibliography"}
            and isinstance(value.get("reason"), str) and bool(value["reason"].strip())
            and isinstance(value.get("source_quote"), str) and bool(value["source_quote"].strip())
            and value["source_quote"] in source)


def semantic_chunks(text: str, limit: int) -> list[str]:
    """Bound requests at sentence/word boundaries, never inside protected tokens."""
    units = re.findall(r"\[\[BOOKFLOW_FOOTNOTE_\d+\]\]|[\u3400-\u9fff]|[^\s\u3400-\u9fff]+\s*|\s+", text)
    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) > limit:
            # Prefer the last sentence boundary, keeping the remainder intact.
            boundaries = list(re.finditer(r"[.!?。！？][\"'’”)]*\s+", current))
            split = boundaries[-1].end() if boundaries and boundaries[-1].end() >= len(current) // 2 else len(current)
            chunks.append(current[:split])
            current = current[split:]
            if current and len(current) + len(unit) > limit:
                chunks.append(current)
                current = ""
        current += unit
    if current:
        chunks.append(current)
    return chunks or [text]


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
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    style: dict[str, Any] = field(default_factory=dict)
    style_origins: dict[str, str] = field(default_factory=dict)
    summary_hashes: dict[str, str] = field(default_factory=dict)
    preparation_fingerprint: str = ""
    book_brief: str = ""
    chapter_briefs: dict[str, str] = field(default_factory=dict)

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
                entries=raw.get("entries", {}),
                style=raw.get("style", {}),
                style_origins=raw.get("style_origins", {}),
                summary_hashes=raw.get("summary_hashes", {}),
                preparation_fingerprint=raw.get("preparation_fingerprint", ""),
                book_brief=raw.get("book_brief", ""),
                chapter_briefs=raw.get("chapter_briefs", {}),
            )
        else:
            memory = cls(path, book_title, context_segments=context_segments, context_chars=context_chars)
        # User configuration is authoritative over model-extracted entries.
        memory.terms = {**memory.terms, **configured_terms}
        memory.locked_terms = set(configured_terms)
        memory.context_segments = context_segments
        memory.context_chars = context_chars
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
            "entries": self.entries,
            "style": self.style,
            "style_origins": self.style_origins,
            "summary_hashes": self.summary_hashes,
            "preparation_fingerprint": self.preparation_fingerprint,
            "book_brief": self.book_brief,
            "chapter_briefs": self.chapter_briefs,
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def fingerprint(self) -> str:
        return _digest({"terms": self.terms, "entries": self.entries, "style": self.style,
                        "summaries": self.chapter_summaries, "book_brief": self.book_brief,
                        "chapter_briefs": self.chapter_briefs, "version": self.version})

    def relevant_terms(self, source_text: str, *, limit: int | None = None) -> dict[str, str]:
        # All applicable terms are mandatory; irrelevant global tails consumed
        # the budget and previously displaced the terms actually needed.
        matches = []
        for source, target in self.terms.items():
            for name in [source, *self.entries.get(source, {}).get("aliases", [])]:
                pattern = re.escape(name)
                if re.search(r"[A-Za-z]", name):
                    pattern = r"(?<!\w)" + pattern + r"(?:s|es|['’]s)?(?!\w)"
                for match in re.finditer(pattern, source_text, re.IGNORECASE):
                    matches.append((self.entries.get(source, {}).get("origin") == "human",
                                    match.end() - match.start(), match.start(), match.end(), source, target))
        occupied: list[tuple[int, int]] = []
        selected = {}
        for _, _, start, end, source, target in sorted(matches, reverse=True):
            if any(start < b and end > a for a, b in occupied):
                continue
            occupied.append((start, end))
            selected[source] = target
        return selected

    def policy_for(self, source_text: str) -> str:
        relevant = self.relevant_terms(source_text)
        entries = []
        for key, target in relevant.items():
            details = {name: self.entries.get(key, {}).get(name) for name in
                       ("meaning", "allowed_variants", "origin") if self.entries.get(key, {}).get(name)}
            entries.append(f"- {key} => {target}" + (" | " + json.dumps(details, ensure_ascii=False) if details else ""))
        return ("Rule priority: human terminology > human style > automatic terminology by sense > automatic style. "
                "Specific locked spellings override any conflicting examples in style prose. "
                "Never replace an approved human spelling with an automatic suggestion. "
                "Numeric unit conversions preserving value are allowed unless explicitly forbidden by the human.\n"
                "Mandatory book style (human overrides are authoritative):\n"
                + json.dumps({"human": {k: v for k, v in self.style.items() if self.style_origins.get(k) == "human"},
                              "automatic": {k: v for k, v in self.style.items() if self.style_origins.get(k) != "human"}}, ensure_ascii=False)
                + "\nLocked terminology (apply by sense; allowed_variants are acceptable):\n"
                + "\n".join(entries))

    def policy_fingerprint(self, source_text: str) -> str:
        return _digest(self.policy_for(source_text))

    def term_violations(self, source_text: str, translation: str) -> dict[str, str]:
        return {source: target for source, target in self.relevant_terms(source_text).items()
                if not any(value.casefold() in translation.casefold() for value in
                           [target, *self.entries.get(source, {}).get("allowed_variants", [])])}

    def context_for(self, *, chapter: str, source_text: str) -> str:
        lines = [
            f"Book title: {self.book_title}",
            f"Current chapter: {chapter}",
            "Use this information only for consistency. Do not translate or repeat this context.",
            self.policy_for(source_text),
        ]
        if self.chapter_summaries:
            summaries = [(key, value) for key, value in self.chapter_summaries.items() if key != chapter][-2:]
            lines.append("Prior chapter summaries:")
            lines.extend(f"- {key}: {value[:500]}" for key, value in summaries)
        if self.recent and self.context_segments:
            lines.append("Recent translated context:")
            recent_chars = 0
            for item in reversed(self.recent[-self.context_segments :]):
                text = item["translation"]
                if recent_chars + len(text) > self.context_chars:
                    text = text[-max(0, self.context_chars - recent_chars):] if self.context_chars > recent_chars else ""
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
    # Source-grounded summaries cover the WHOLE chapter. Translation is not
    # evidence of facts; retaining its hash elsewhere invalidates summaries
    # after edits without amplifying an initial mistranslation.
    parts = semantic_chunks(source_text, 5000)
    summaries = []
    for part in parts:
        prompt = (
            "Write a concise continuity summary grounded ONLY in this original book passage. "
            "Include people, referents, concepts, argument/events, and voice. Do not assume fiction. "
            "Do not invent facts. Return plain text in at most 500 characters.\n"
            f"Book: {title}\nChapter: {chapter}\nOriginal:\n{part}"
        )
        summaries.append(client.chat("", prompt).strip())
    while len(summaries) > 1:
        combined = []
        for start in range(0, len(summaries), 6):
            prompt = ("Merge these sequential chapter notes into one factual continuity summary in at most 700 characters. "
                      "Retain important referents, concepts, argument and events from beginning through end. "
                      "Do not add facts.\n" + "\n".join(summaries[start:start + 6]))
            combined.append(client.chat("", prompt).strip())
        summaries = combined
    return summaries[0] if summaries else ""
