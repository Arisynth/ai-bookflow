from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        if value.startswith("[") or value.startswith("{"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        return value


def _minimal_yaml(text: str) -> dict[str, Any]:
    """Small fallback parser for the simple nested config we ship.

    PyYAML is recommended, but keeping this fallback makes the CLI usable in a
    clean Python installation without turning configuration into a hard runtime
    dependency.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if content.startswith("-"):
            parent = stack[-1][1]
            values = parent.setdefault("__list__", [])
            values.append(_scalar(content[1:].strip()))
            continue
        if ":" not in content:
            raise ValueError(f"Unsupported config line: {raw}")
        key, value = content.split(":", 1)
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key.strip()] = _scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"__list__"}:
                return value["__list__"]
            return {k: clean(v) for k, v in value.items()}
        return value

    return clean(root)


def load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(text)
        if isinstance(value, dict):
            return value
    except ImportError:
        pass
    return _minimal_yaml(text)


@dataclass
class Config:
    workspace: Path = Path.home() / ".bookflow"
    ollama_url: str = "http://127.0.0.1:11434"
    translation_model: str = ""
    review_model: str = ""
    source_language: str | None = None
    target_language: str = "zh-CN"
    output_mode: str = "bilingual"
    output_formats: list[str] = field(default_factory=lambda: ["epub"])
    marker_command: str = "marker_single"
    calibre_command: str = "ebook-convert"
    babelebook_command: str = "babel-ebook"
    pdf_engine: str = "marker"
    ocr_mode: str = "auto"
    marker_page_range: str | None = None
    translation_engine: str = "ollama"
    chunk_size: int = 1200
    concurrency: int = 1
    request_retries: int = 3
    retry_backoff_seconds: float = 2.0
    glossary: dict[str, str] = field(default_factory=dict)
    review_enabled: bool = True
    max_revision_rounds: int = 1
    minor_revision_rounds: int = 2
    serious_revision_rounds: int = 5
    strict_review: bool = False
    review_num_ctx: int | None = 16384
    review_num_predict: int | None = 4096
    review_think: bool | None = False
    dispute_enabled: bool = True
    dispute_think: bool | None = None
    temperature: float = 0.2
    top_p: float | None = None
    num_ctx: int | None = None
    think: bool | None = False
    prompt_version: str = "bookflow-v3-footnotes"
    validate_epub: bool = True
    consistency_enabled: bool = True
    term_extraction_enabled: bool = True
    chapter_summary_enabled: bool = True
    context_segments: int = 2
    context_chars: int = 3000
    term_extraction_chars: int = 18000
    style: dict[str, Any] = field(default_factory=dict)
    terminology: dict[str, Any] = field(default_factory=dict)
    chapter_review_enabled: bool = True
    chapter_review_chars: int = 6000
    book_revision_rounds: int = 2
    epubcheck_command: str = "epubcheck"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "Config":
        runtime = raw.get("runtime", {}) or {}
        models = raw.get("models", {}) or {}
        processing = raw.get("processing", {}) or {}
        pdf = raw.get("pdf", {}) or {}
        translation = raw.get("translation", {}) or {}
        review = raw.get("review", {}) or {}
        output = raw.get("output", {}) or {}
        params = translation.get("parameters", {}) or {}
        return cls(
            workspace=Path(runtime.get("workspace", Path.home() / ".bookflow")).expanduser(),
            ollama_url=str(runtime.get("ollama_url", cls.ollama_url)).rstrip("/"),
            translation_model=str(models.get("translation", "")),
            review_model=str(models.get("review", "")),
            source_language=processing.get("source_language"),
            target_language=str(processing.get("target_language", cls.target_language)),
            output_mode=str(processing.get("output_mode", cls.output_mode)),
            output_formats=list(output.get("formats", ["epub"])),
            marker_command=str(pdf.get("marker_command", cls.marker_command)),
            calibre_command=str(raw.get("calibre", {}).get("command", cls.calibre_command)),
            babelebook_command=str(translation.get("babelebook_command", cls.babelebook_command)),
            pdf_engine=str(pdf.get("engine", cls.pdf_engine)),
            ocr_mode=str(pdf.get("ocr_mode", cls.ocr_mode)),
            marker_page_range=pdf.get("page_range"),
            translation_engine=str(translation.get("engine", cls.translation_engine)),
            chunk_size=int(translation.get("chunk_size", cls.chunk_size)),
            concurrency=int(translation.get("concurrency", cls.concurrency)),
            request_retries=int(translation.get("request_retries", cls.request_retries)),
            retry_backoff_seconds=float(translation.get("retry_backoff_seconds", cls.retry_backoff_seconds)),
            glossary=dict(translation.get("glossary", {}) or {}),
            review_enabled=bool(review.get("enabled", cls.review_enabled)),
            max_revision_rounds=int(review.get("max_revision_rounds", cls.max_revision_rounds)),
            minor_revision_rounds=int(review.get("minor_revision_rounds", review.get("max_revision_rounds", cls.minor_revision_rounds))),
            serious_revision_rounds=int(review.get("serious_revision_rounds", cls.serious_revision_rounds)),
            strict_review=bool(review.get("strict", cls.strict_review)),
            review_num_ctx=review.get("num_ctx", cls.review_num_ctx),
            review_num_predict=review.get("num_predict", cls.review_num_predict),
            review_think=review.get("think", cls.review_think),
            dispute_enabled=bool(review.get("dispute_enabled", cls.dispute_enabled)),
            dispute_think=review.get("dispute_think", cls.dispute_think),
            temperature=float(params.get("temperature", translation.get("temperature", cls.temperature))),
            top_p=params.get("top_p", translation.get("top_p")),
            num_ctx=params.get("num_ctx", translation.get("num_ctx")),
            think=params.get("think", translation.get("think", cls.think)),
            prompt_version=str(translation.get("prompt_version", cls.prompt_version)),
            validate_epub=bool(output.get("validate_epub", cls.validate_epub)),
            consistency_enabled=bool(translation.get("consistency_enabled", cls.consistency_enabled)),
            term_extraction_enabled=bool(translation.get("term_extraction_enabled", cls.term_extraction_enabled)),
            chapter_summary_enabled=bool(translation.get("chapter_summary_enabled", cls.chapter_summary_enabled)),
            context_segments=int(translation.get("context_segments", cls.context_segments)),
            context_chars=int(translation.get("context_chars", cls.context_chars)),
            term_extraction_chars=int(translation.get("term_extraction_chars", cls.term_extraction_chars)),
            style=({"instructions": translation["style"]} if isinstance(translation.get("style"), str)
                   else dict(translation.get("style", {}) or {})),
            terminology=dict(translation.get("terminology", {}) or {}),
            chapter_review_enabled=bool(review.get("chapter_enabled", cls.chapter_review_enabled)),
            chapter_review_chars=int(review.get("chapter_chars", cls.chapter_review_chars)),
            book_revision_rounds=int(review.get("book_revision_rounds", cls.book_revision_rounds)),
            epubcheck_command=str(output.get("epubcheck_command", cls.epubcheck_command)),
        )

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        return cls.from_mapping(load_mapping(path))

    def validate(self) -> None:
        if self.output_mode not in {"original", "translated", "bilingual"}:
            raise ValueError("output_mode must be original, translated, or bilingual")
        if self.ocr_mode not in {"auto", "force", "off"}:
            raise ValueError("ocr_mode must be auto, force, or off")
        if self.translation_engine not in {"ollama", "babelebook"}:
            raise ValueError("translation.engine must be ollama or babelebook")
        if self.chunk_size < 100:
            raise ValueError("translation.chunk_size must be at least 100")
        if self.request_retries < 0:
            raise ValueError("translation.request_retries must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("translation.retry_backoff_seconds must be non-negative")
        if self.max_revision_rounds < 0:
            raise ValueError("review.max_revision_rounds must be non-negative")
        if self.minor_revision_rounds < 0:
            raise ValueError("review.minor_revision_rounds must be non-negative")
        if self.serious_revision_rounds < self.minor_revision_rounds:
            raise ValueError("review.serious_revision_rounds must be >= minor_revision_rounds")
        if self.review_num_ctx is not None and int(self.review_num_ctx) < 256:
            raise ValueError("review.num_ctx must be at least 256")
        if self.review_num_predict is not None and int(self.review_num_predict) < 32:
            raise ValueError("review.num_predict must be at least 32")
        if not self.output_formats:
            raise ValueError("output.formats must contain at least epub")
        if self.context_segments < 0 or self.context_chars < 0:
            raise ValueError("translation context budgets must be non-negative")
        if self.term_extraction_chars < 500 or self.chapter_review_chars < 500:
            raise ValueError("preparation/review character budgets must be at least 500")
        if self.book_revision_rounds < 0:
            raise ValueError("review.book_revision_rounds must be non-negative")
        if not all(isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
                   for k, v in self.glossary.items()):
            raise ValueError("translation.glossary must map nonempty source strings to target strings")
        for source, entry in self.terminology.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("target"), str) or not entry["target"].strip():
                raise ValueError(f"translation.terminology.{source} requires a target")
            for key in ("aliases", "allowed_variants"):
                if key in entry and (not isinstance(entry[key], list) or not all(isinstance(x, str) for x in entry[key])):
                    raise ValueError(f"terminology {key} must be a list of strings")
