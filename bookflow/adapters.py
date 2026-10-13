from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .errors import ExternalToolError
from .epub import Segment
from .memory import semantic_chunks, valid_term_exception, term_occurs


def executable_available(command: str) -> bool:
    return bool(shutil.which(command))


def run_command(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise ExternalToolError(
            f"找不到外部工具 {args[0]}。请按 README 安装，或在 config.yaml 配置正确的命令路径。"
        ) from exc
    if result.returncode:
        details = (result.stderr or result.stdout).strip()[-4000:]
        raise ExternalToolError(f"外部工具执行失败 ({result.returncode}): {' '.join(args)}\n{details}")
    return result


@dataclass
class MarkerAdapter:
    command: str

    def convert(self, source: Path, output_dir: Path, ocr_mode: str, page_range: str | None = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        args = [self.command, str(source), "--output_format", "markdown", "--output_dir", str(output_dir)]
        if page_range:
            args.extend(["--page_range", page_range])
        if ocr_mode == "force":
            args.append("--force_ocr")
        elif ocr_mode == "off":
            args.append("--disable_ocr")
        run_command(args)
        markdown = sorted(output_dir.rglob("*.md"))
        if not markdown:
            raise ExternalToolError(f"Marker 执行成功但未生成 Markdown: {output_dir}")
        return markdown[0]


@dataclass
class CalibreAdapter:
    command: str

    def convert(self, source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        args = [self.command, str(source), str(destination)]
        # The XHTML already contains a structured visible TOC. Avoid Calibre's
        # generated inline TOC for Kindle outputs, which duplicates the TOC and
        # can make EPUB2 round-trips invalid.
        if destination.suffix.lower() in {".azw3", ".mobi"}:
            args.append("--no-inline-toc")
        run_command(args)
        if not destination.exists():
            raise ExternalToolError(f"Calibre 执行成功但未生成输出: {destination}")
        return destination


@dataclass
class BabelEbookAdapter:
    """Adapter for the selected BabelEbook CLI.

    BabelEbook owns EPUB-aware bilingual layout. BookFlow uses the native
    Ollama adapter by default because BabelEbook has no macOS binary and its
    Rust CLI must be built locally. This adapter keeps that integration behind
    the same provider boundary for users who install it.
    """

    command: str

    def translate(self, source: Path, destination: Path, config: Config) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # BabelEbook 0.5 exposes its current interface through the `translate`
        # subcommand and a JSON config file; older releases accepted `--mode`
        # directly. Generate the current config so the adapter matches the
        # installed CLI while keeping the Python boundary stable.
        config_path = destination.parent / f".{destination.stem}.babelebook.json"
        babel_config = {
            "source": str(source),
            "output": str(destination),
            "provider": "ollama",
            "base_url": config.ollama_url,
            "model": config.translation_model,
            "concurrency": config.concurrency,
            "target_lang": config.target_language,
            "temperature": config.temperature,
            "output_mode": "translation_only" if config.output_mode == "translated" else "bilingual",
            "cache_dir": str(destination.parent / ".babel_ebook_cache"),
            "checkpoint_dir": str(destination.parent / ".babel_ebook_checkpoints"),
        }
        config_path.write_text(json.dumps(babel_config, ensure_ascii=False, indent=2), encoding="utf-8")
        args = [
            self.command,
            "translate",
            str(source),
            "-o",
            str(destination),
            "--config",
            str(config_path),
        ]
        try:
            run_command(args)
            if not destination.exists():
                raise ExternalToolError(f"BabelEbook 执行成功但未生成输出: {destination}")
        finally:
            config_path.unlink(missing_ok=True)
        return destination


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        opener: Callable[..., Any] | None = None,
        options: dict[str, Any] | None = None,
        think: bool | None = None,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        if opener is not None:
            self.opener = opener
        elif urllib.parse.urlparse(self.base_url).hostname in {"127.0.0.1", "localhost", "::1"}:
            # A desktop Ollama daemon is local; do not accidentally route its
            # API calls through a stale system proxy.
            self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        else:
            self.opener = urllib.request.urlopen
        self.options = options or {}
        self.think = think
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))

    def chat(self, system: str, user: str, *, json_format: bool | dict[str, Any] = False) -> str:
        if not self.model:
            raise ExternalToolError("未配置 models.translation；请在 config.yaml 中填写本地 Ollama 模型名称")
        messages = [{"role": "user", "content": user}]
        if system.strip():
            messages.insert(0, {"role": "system", "content": system})
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": messages,
        }
        if self.options:
            payload["options"] = dict(self.options)
        # Budget the complete prompt before sending it. Never silently slice
        # book rules or source text. Grow only this request, with a local cap.
        prompt = system + user
        non_ascii = sum(ord(char) > 127 for char in prompt)
        estimate = non_ascii * 2 + (len(prompt) - non_ascii + 2) // 3 + 256
        output_budget = max(1024, self.options.get("num_predict", 2048))
        needed = estimate + output_budget
        configured = int(self.options.get("num_ctx") or 8192)
        if needed > configured:
            expanded = ((needed + 1023) // 1024) * 1024
            if expanded > max(configured, 32768):
                raise ExternalToolError("完整请求超过本地上下文预算；请减小分段/章节审校批次，未截断或发送正文")
            payload.setdefault("options", {})["num_ctx"] = expanded
        if self.think is not None:
            payload["think"] = self.think
        if json_format:
            payload["format"] = json_format if isinstance(json_format, dict) else "json"
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        open_request = self.opener.open if hasattr(self.opener, "open") else self.opener
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with open_request(request, timeout=600) as response:
                    body = json.loads(response.read().decode("utf-8"))
                if body.get("done_reason") == "length":
                    # Never persist a partial translation as a successful reply.
                    # Retry with a larger per-request output budget, leaving the
                    # caller's configured default untouched.
                    budget = payload.get("options", {}).get("num_predict")
                    if isinstance(budget, int) and budget > 0:
                        payload["options"] = {**payload.get("options", {}), "num_predict": budget * 2}
                        request.data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    raise ExternalToolError("Ollama 输出被截断 (done_reason=length)")
                try:
                    return str(body["message"]["content"]).strip()
                except (KeyError, TypeError) as exc:
                    raise ExternalToolError(f"Ollama 响应缺少 message.content: {body}") from exc
            except urllib.error.HTTPError as exc:
                last_error = exc
                retryable = exc.code == 408 or exc.code == 429 or exc.code >= 500
                if not retryable:
                    raise ExternalToolError(f"Ollama HTTP {exc.code}: {exc.reason}") from exc
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ExternalToolError) as exc:
                last_error = exc
            if attempt >= self.max_retries:
                break
            delay = self.retry_backoff_seconds * (2**attempt)
            if delay:
                time.sleep(delay)
        if isinstance(last_error, json.JSONDecodeError):
            raise ExternalToolError("Ollama 返回了非 JSON 响应（重试次数已用尽）") from last_error
        raise ExternalToolError(
            f"无法连接 Ollama {self.base_url}；已重试 {self.max_retries} 次。"
            f" 请启动 Ollama 并确认模型已下载: {last_error}"
        ) from last_error


class OllamaTranslator:
    def __init__(self, client: OllamaClient, config: Config):
        self.client = client
        self.config = config

    def revise(
        self, segment: Segment, source_language: str, context: str,
        initial_translation: str, previous_translation: str,
        history: list[dict[str, Any]], *, rewrite: bool = False,
    ) -> str:
        """Revise one complete segment in one call, without chunking or slicing feedback."""
        system = (
            f"You edit a translation from {source_language} to {self.config.target_language}. "
            "Return only the COMPLETE corrected translation of the Original, exactly once. "
            "Compare the latest feedback with the actual previous translation and original. "
            "Fix every substantiated remaining error; retain correct prior fixes. Historical feedback "
            "must not undo a verified correction. Change only wording required by the current substantiated defects; "
            "do not replace already-correct sentences merely for variety. Historical feedback "
            "is fallible: do not reintroduce resolved errors or follow suggestions that contradict the source. "
            "Do not translate the context or concatenate alternative versions. Never add explanations. "
            "Preserve numbers, names and every [[BOOKFLOW_FOOTNOTE_N]] and [[BOOKFLOW_IMAGE_N]] placeholder exactly and in order. "
            "Use natural target-language punctuation, not source-language grammar rules. "
            "Before returning, silently check that each requested fix appears in the actual output."
        )
        user = (
            f"Book context:\n{context}\n\nGlossary:\n"
            f"{json.dumps({k: v for k, v in self.config.glossary.items() if term_occurs(k, segment.text)}, ensure_ascii=False)}\n\n"
            f"Initial translation (round 0):\n{initial_translation}\n\n"
            f"Previous candidate translation:\n{previous_translation}\n\n"
            "Complete review feedback history (entry 0 is Initial review (round 0)):\n"
            f"{json.dumps([{k: v for k, v in entry.items() if k != 'preliminary_review'} for entry in history], ensure_ascii=False)}\n\n"
            f"Current correction request:\n{history[-1].get('suggestion', '')}\n\n"
            "If local edits are insufficient, completely reorganize the affected sentences. "
            + ("The previous attempt made no progress. Rewrite the paragraph afresh from the source, "
               "using the feedback as a checklist.\n\n" if rewrite else "\n\n")
            + f"Original (the only text to translate):\n{segment.text}"
        )
        # Deliberately bypass translate(): its chunk_size applies to initial
        # translation only, never to a whole-paragraph editing request.
        return self.client.chat(system, user).strip()

    def translate(self, segment: Segment, source_language: str, context: str) -> str:
        text = segment.text
        if text in self.config.glossary:
            return self.config.glossary[text]
        if not re.sub(r"\[\[BOOKFLOW_(?:FOOTNOTE|IMAGE)_\d+\]\]", "", text).strip():
            return text
        # Short structural headings are not useful language-model work and
        # can cause small translation models to continue with unrelated text.
        # Normalize the common English book heading form deterministically so
        # the EPUB structure is stable and the model only sees prose.
        heading = re.fullmatch(r"\s*Chapter\s+(\d+)\s*", text, flags=re.IGNORECASE)
        if heading and source_language == "en" and self.config.target_language.lower() in {"zh-cn", "zh-hans"}:
            return f"第{heading.group(1)}章"
        chunks = semantic_chunks(text, self.config.chunk_size)
        glossary = "；".join(f"{k}={v}" for k, v in self.config.glossary.items() if term_occurs(k, text)) or "无"
        language_name = {"en": "English", "de": "German", "fr": "French", "es": "Spanish"}.get(source_language, source_language)
        translated: list[str] = []
        for chunk in chunks:
            if self.client.model.lower().startswith("translategemma"):
                # TranslateGemma's official Ollama prompt guide specifies one
                # user message and this explicit source/target format. Keep
                # the prompt exact: extra context made short headings prone to
                # hallucination in local evaluation.
                system = ""
                user = (
                    f"You are a professional {language_name} ({source_language}) to Simplified Chinese (zh-Hans) translator. "
                    "Your goal is to accurately convey the meaning and nuances of the original text while adhering to Simplified Chinese grammar, vocabulary, and cultural sensitivities. "
                    "Produce only the Simplified Chinese translation, without any additional explanations or commentary. "
                    f"Mandatory book policy and context (do not translate):\n{context}\nGlossary: {glossary}\n"
                    f"Please translate the following {source_language} text into Simplified Chinese:\n\n\n{chunk}"
                )
            else:
                context_for_prompt = context
                system = (
                    "You are a careful book translator. Follow the supplied book style and terminology. "
                    "Translate faithfully without summarizing, adding, or omitting. "
                    f"Translate from {source_language} to {self.config.target_language}. Preserve numbers, names, punctuation, "
                    "and paragraph meaning. Return only the translation."
                )
                user = (
                    f"Book context:\n{context_for_prompt}\n\nGlossary:\n{glossary}\n\n"
                    + (f"Previous translated part of THIS paragraph (continuity only):\n{translated[-1][-1000:]}\n\n" if translated else "")
                    + f"Paragraph to translate:\n{chunk}"
                )
            if "[[BOOKFLOW_" in chunk:
                instruction = (
                    " Preserve every [[BOOKFLOW_FOOTNOTE_N]] and [[BOOKFLOW_IMAGE_N]] placeholder exactly, including its spelling and relative position; "
                    "do not translate, renumber, remove, or explain these placeholders."
                )
                system += instruction
                user += "\n\nFootnote placeholder rule: keep every [[BOOKFLOW_FOOTNOTE_N]] and [[BOOKFLOW_IMAGE_N]] exactly as written."
            if self.client.model.lower().startswith("translategemma"):
                result = self.client.chat(system, user)
                # TranslateGemma can occasionally copy an English adjective
                # into an otherwise Chinese paragraph. Retry just that chunk
                # with a compact repair hint instead of silently accepting a
                # mixed-language result.
                residual = sorted(set(re.findall(r"\b[A-Za-z]{4,}\b", result)))
                if residual:
                    hints = {
                        "tireless": "不知疲倦的",
                        "honorable": "体面的",
                        "honourable": "体面的",
                    }
                    hint_lines = [f"{word} = {hints[word.lower()]}" for word in residual if word.lower() in hints]
                    repair = (
                        f"You are a professional {language_name} ({source_language}) to Simplified Chinese (zh-Hans) translator. "
                        "Produce only the Simplified Chinese translation. Translate every source word; do not leave English words untranslated."
                    )
                    if hint_lines:
                        repair += " Use these required translations: " + "; ".join(hint_lines) + "."
                    repair += f" Mandatory book policy and glossary:\n{context}\n{glossary}\nPlease translate the following {source_language} text into Simplified Chinese:\n\n\n{chunk}"
                    result = self.client.chat("", repair)
                translated.append(result)
            else:
                translated.append(self.client.chat(system, user))
        return ("" if self.config.target_language.lower().startswith("zh") else " ").join(translated).strip()


class OllamaReviewer:
    def __init__(self, client: OllamaClient, *, max_parse_retries: int = 2, verifier: Any = None):
        self.client = client
        self.verifier = verifier
        # Transport retries are handled by OllamaClient. These retries are
        # separate: the request succeeded, but the model violated the review
        # response contract (for example, truncated JSON or a code fence).
        self.max_parse_retries = max(0, int(max_parse_retries))

    def review(self, original: str, translated: str, context: str, *,
               _allegations: dict[str, Any] | None = None) -> dict[str, Any]:
        system = (
            "You review a translation concisely. Return only one complete JSON object with four required keys: "
            "ok (boolean), severity (string), issues (array of short strings), and suggestion (string). "
            "severity must be one of none, minor, major, or critical. Use none only when ok=true. "
            "Use minor for typography, punctuation, stylistic, or low-impact terminology issues; "
            "major for omissions, mistranslations, missing numbers, or meaning changes; "
            "critical only for severe omissions or reversed central meaning, never merely because retries failed. "
            "Assess only the current Translation against Original. Quote exact problematic text in each issue "
            "and give a specific replacement or missing source phrase. Do not invent defects. "
            "Do not flag acceptable synonyms or optional stylistic preferences. Chinese translations do not "
            "need English articles such as 'the'; translate names and titles naturally. "
            "Meaning can appear in a different sentence position. A nonliteral but equivalent expression "
            "is not an omission. Natural punctuation changes are allowed. Do not fill a quota of issues. "
            "Set ok=true only when issues is []. If there is any issue, set ok=false. "
            "suggestion must be a concise, actionable correction plan that addresses every issue, "
            "but must not be a rewritten paragraph. If ok=true, suggestion must be empty. "
            "Do not write analysis, Markdown fences, or any text outside the JSON."
            " The supplied book style and locked terminology apply to review and revision alike. "
            "Use adjacent SOURCE paragraphs to resolve references and meaning; never judge just an isolated phrase. "
            "Check number ownership, signs, units, percentages, negation, omissions and additions, not only digit presence. "
            "Optional terminology_exceptions may map an AUTOMATIC term to {kind: different_sense or bibliography, "
            "source_quote: exact original evidence, reason: precise justification}. Never waive a human term rule "
            "or use generic synonym preference as an exception. Do not repeat resolved allegations."
        )
        user = f"Context:\n{context}\n\nOriginal:\n{original}\n\nTranslation:\n{translated}"
        if _allegations is not None:
            system += (
                " You are now checking whether a preliminary rejection is justified. Its allegations "
                "are NOT facts. Inspect the supplied CURRENT translation for evidence. Reject an "
                "allegation if the meaning is already expressed anywhere, is an acceptable paraphrase, "
                "or requests an optional stylistic edit. Example: '这是一回事' already expresses "
                "'It is one thing', and '再往前一年也是如此' already expresses 'And the year before that'. "
                "Keep genuine mistranslations, untranslated words, omissions, additions or repetitions. "
                "For each confirmed issue quote the actual source and translation evidence and specify "
                "the semantic defect. Assign severity by meaning impact, never by retry count. "
                "Return ok=true with no issues if no genuine defects remain."
            )
            user += "\n\nPreliminary allegations to VERIFY, not repeat:\n" + json.dumps(_allegations, ensure_ascii=False)
        response_schema = {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "severity": {"type": "string", "enum": ["none", "minor", "major", "critical"]},
                "issues": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 160},
                    "maxItems": 3,
                },
                "suggestion": {"type": "string", "maxLength": 800},
                "terminology_exceptions": {"type": "object", "additionalProperties": {
                    "type": "object", "properties": {"kind": {"type": "string", "enum": ["different_sense", "bibliography"]},
                    "source_quote": {"type": "string"}, "reason": {"type": "string"}},
                    "required": ["kind", "source_quote", "reason"], "additionalProperties": False}},
            },
            "required": ["ok", "severity", "issues", "suggestion"],
            "additionalProperties": False,
        }
        last_raw = ""
        for attempt in range(self.max_parse_retries + 1):
            # A malformed response is often a generation-limit truncation.
            # Give later format retries a larger bounded budget instead of
            # repeating the same too-small request.
            options = getattr(self.client, "options", None)
            if attempt and isinstance(options, dict) and isinstance(options.get("num_predict"), int):
                options["num_predict"] = min(options["num_predict"] * 2, 8192)
            prompt = user
            if attempt:
                prompt += (
                    "\n\nYour previous response did not follow the required JSON contract. "
                    "Retry now. Return exactly one complete JSON object, with no Markdown fences, "
                    "no commentary, with exactly these keys and types: ok boolean, severity string, "
                    "issues array of strings, and suggestion string. Set ok=true only with issues=[] "
                    "and suggestion=\"\". Do not rewrite the paragraph."
                )
            last_raw = self.client.chat(system, prompt, json_format=response_schema)
            try:
                value = json.loads(last_raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            if not isinstance(value.get("ok"), bool):
                continue
            severity = value.get("severity")
            if severity is None:
                # Accept old reviewer checkpoints/models while migrating
                # to severity-aware review responses.
                severity = "none" if value["ok"] else "major"
            if severity not in {"none", "minor", "major", "critical"}:
                continue
            if value["ok"] and severity != "none":
                continue
            if not value["ok"] and severity == "none":
                severity = "major"
            if not isinstance(value.get("issues"), list) or not all(
                isinstance(issue, str) for issue in value["issues"]
            ):
                continue
            suggestion = value.get("suggestion", "")
            if not isinstance(suggestion, str):
                continue
            if value["ok"] and value["issues"]:
                continue
            if not value["ok"] and not value["issues"]:
                continue
            if value["ok"] and suggestion.strip():
                continue
            result = {
                "ok": value["ok"],
                "severity": severity,
                "issues": value["issues"],
                # Keep the suggestion actionable but prevent the reviewer from
                # generating a second full translation in the review call.
                "suggestion": (suggestion or "；".join(value["issues"])) if not value["ok"] else "",
            }
            exceptions = value.get("terminology_exceptions", {})
            if exceptions and isinstance(exceptions, dict):
                result["terminology_exceptions"] = {k: v for k, v in exceptions.items()
                                                    if valid_term_exception(v, original)}
            if not result["ok"] and _allegations is None:
                try:
                    confirmed = (self.verifier or self).review(original, translated, context, _allegations=result)
                except ExternalToolError as exc:
                    if not self.verifier:
                        raise
                    # An unavailable reasoning mode is not a required manual
                    # configuration step. Fall back to the ordinary evidence
                    # verifier, and preserve the failure in the audit record.
                    confirmed = self.review(original, translated, context, _allegations=result)
                    confirmed["verification_error"] = str(exc)
                confirmed["preliminary_review"] = result
                return confirmed
            return result
        return {
            "ok": False,
            "severity": "major",
            "issues": ["审校模型未返回合法 JSON"],
            # Never feed an invalid model response into the revision prompt.
            "suggestion": "",
            "review_error": "invalid_json",
            "raw_response": last_raw,
        }

    def select_best(
        self,
        original: str,
        context: str,
        candidates: list[str],
        feedback_history: str,
    ) -> dict[str, Any]:
        """Compare complete candidates pairwise so long paragraphs are never clipped."""
        if len(candidates) > 2:
            winner = 0
            comparisons = []
            for challenger in range(1, len(candidates)):
                choice = self.select_best(original, context, [candidates[winner], candidates[challenger]], "")
                comparisons.append({"indices": [winner, challenger], **choice})
                if choice.get("selection_error"):
                    return {"selected_index": winner, "reason": choice["reason"],
                            "selection_error": choice["selection_error"], "comparisons": comparisons}
                if choice["selected_index"] == 1:
                    winner = challenger
            last_pair = comparisons[-1]["indices"]
            return {"selected_index": winner,
                    "reason": f"Final pair: local 0 = candidate {last_pair[0]}, local 1 = candidate {last_pair[1]}. " + comparisons[-1]["reason"],
                    "comparisons": comparisons}
        system = (
            "You are the final translation quality judge. Choose the single best candidate translation. "
            "Return only one complete JSON object with exactly two keys: selected_index (integer) and "
            "reason (short string). Prefer faithful, complete, natural Chinese; preserve numbers, names, "
            "punctuation, and footnote placeholders. Judge the actual complete text, not historical claims. "
            "Quote a concrete difference in the reason; do not claim a fix absent from the selected text. "
            "Do not rewrite any candidate. Choosing the best does not mean it is error-free."
        )
        candidate_text = "\n\n".join(
            f"--- Candidate {index} ---\n{text}" for index, text in enumerate(candidates)
        )
        user = (
            f"Context:\n{context}\n\nOriginal:\n{original}\n\n{candidate_text}"
        )
        schema = {
            "type": "object",
            "properties": {
                "selected_index": {"type": "integer", "minimum": 0, "maximum": max(0, len(candidates) - 1)},
                "reason": {"type": "string", "maxLength": 300},
            },
            "required": ["selected_index", "reason"],
            "additionalProperties": False,
        }
        last_raw = ""
        for attempt in range(self.max_parse_retries + 1):
            prompt = user
            if attempt:
                prompt += (
                    "\n\nReturn only valid JSON with selected_index and reason. "
                    "selected_index must refer to one of the candidates."
                )
            last_raw = self.client.chat(system, prompt, json_format=schema)
            try:
                value = json.loads(last_raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            index = value.get("selected_index")
            if type(index) is not int or not 0 <= index < len(candidates):
                continue
            if not isinstance(value.get("reason"), str):
                continue
            return {"selected_index": index, "reason": value["reason"]}
        return {
            "selected_index": 0,
            "reason": "审校模型未返回合法的候选选择 JSON",
            "selection_error": "invalid_json",
            "raw_response": last_raw,
        }
