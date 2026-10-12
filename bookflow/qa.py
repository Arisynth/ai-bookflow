from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .epub import Segment


@dataclass
class QAReport:
    ok: bool
    checked: int
    issues: list[str]

    def as_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "checked": self.checked, "issues": self.issues}


def deterministic_qa(segments: list[Segment], translations: dict[str, str]) -> QAReport:
    issues: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        if segment.segment_id in seen:
            issues.append(f"重复段落 ID: {segment.segment_id}")
        seen.add(segment.segment_id)
        translated = translations.get(segment.segment_id, "").strip()
        if not translated:
            issues.append(f"段落无译文: {segment.segment_id}")
            continue
        original_numbers = _numeric_values(segment.text)
        translated_numbers = _numeric_values(translated)
        missing = original_numbers - translated_numbers
        if missing:
            issues.append(f"段落 {segment.segment_id} 数字可能变化/丢失: {sorted(missing)}")
    return QAReport(not issues, len(segments), issues)


def _numeric_values(text: str) -> set[str]:
    """Normalize explicit scale units: 1.3 billion and 13 亿 have the same value."""
    text = re.sub(r"\[\[BOOKFLOW_FOOTNOTE_\d+\]\]", "", text)
    # Digits in bibliographic URLs are identifiers, not translatable numbers.
    # EPUBs often insert zero-width characters into the source URL, which can
    # otherwise make the same URL look like a different set of numbers.
    text = re.sub(r"(?:https?://|www\.)[^\s，。；！？、）》）】》]+", "", text, flags=re.IGNORECASE)
    # Common lexical labels rather than quantities (course title and
    # presidential nickname). Their localized Chinese forms often omit the
    # English number while preserving the meaning.
    text = re.sub(r"\bStatistics\s+101\b|\bBush\s+41\b", "", text, flags=re.IGNORECASE)
    scales = {"thousand": 1000, "million": 1000000, "billion": 1000000000,
              "trillion": 1000000000000,
              "千": 1000, "万": 10000, "百万": 1000000, "千万": 10000000,
              "亿": 100000000, "十亿": 1000000000}
    values: set[str] = set()
    compound_pattern = r"(?<![\d.])(?P<number>\d+(?:\.\d+)?)(?:\s*)(?P<scale>万亿美元|亿美元|万英镑)"
    for match in re.finditer(compound_pattern, text):
        scale = {"万亿美元": 1000000000000, "亿美元": 100000000, "万英镑": 1000000000000}[match["scale"]]
        values.add(_number_key(str(Decimal(match["number"]) * scale)))
        # Remove this exact span from the general pass below so that 2万亿
        # is not also interpreted as the unrelated number 2 or 2万.
        text = text.replace(match.group(0), " " * len(match.group(0)), 1)
    pattern = r"(?<![\d.])(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)(?:\s*(?P<scale>thousand\b|million\b|billion\b|trillion\b|十亿|百万|千万|亿|万|千))?"
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        value = Decimal(match["number"].replace(",", ""))
        value *= scales.get((match["scale"] or "").lower(), 1)
        values.add(_number_key(str(value)))
    # Bibliographic page ranges are frequently expanded by translation:
    # 158–62 and 158–162 denote the same pages. Add both representations so
    # formatting expansion does not look like a lost number.
    for match in re.finditer(r"(?P<start>\d{2,4})\s*[–-]\s*(?P<end>\d{1,4})", text):
        start = match["start"]
        end = match["end"]
        values.add(_number_key(end))
        if len(end) < len(start):
            expanded = start[: len(start) - len(end)] + end
            values.add(_number_key(expanded))
        elif len(end) == len(start) and len(end) >= 3:
            values.add(_number_key(end[-2:]))
    values.update(_chinese_numeric_values(text))
    return values


_CHINESE_DIGITS = {
    "零": 0, "〇": 0, "○": 0, "一": 1, "二": 2, "两": 2,
    "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}
_CHINESE_NUMBER_RE = r"[零〇○一二两三四五六七八九十百千万亿]+"


def _parse_chinese_number(token: str) -> int | None:
    """Parse common Chinese cardinal numbers, including 一亿 and 二十六."""
    if not token:
        return None
    total = 0
    section = 0
    digit: int | None = None
    for char in token:
        if char in _CHINESE_DIGITS:
            digit = _CHINESE_DIGITS[char]
            continue
        unit = _CHINESE_UNITS.get(char)
        if unit is None:
            return None
        if unit < 10000:
            section += (digit if digit is not None else 1) * unit
        else:
            section += digit or 0
            if section == 0:
                section = 1
            total += section * unit
            section = 0
        digit = None
    return total + section + (digit or 0)


def _chinese_numeric_values(text: str) -> set[str]:
    """Recognize Chinese numerals in numeric contexts without treating every 一 as a number.

    The source is normally English while the translation is Chinese, so a
    literal Arabic-digit comparison misses forms such as ``百分之百``,
    ``一亿分之一``, ``第一章`` and ``系统二``.  Restricting extraction to
    classifiers and explicit ordinal/system/fraction contexts avoids turning
    ordinary words such as ``一定`` into numeric evidence.
    """
    values: set[str] = set()

    def add(token: str) -> None:
        parsed = _parse_chinese_number(token)
        if parsed is not None:
            values.add(_number_key(str(parsed)))

    # Chinese fraction notation is denominator-before-numerator: 一亿分之一.
    for match in re.finditer(
        rf"(?P<den>{_CHINESE_NUMBER_RE})分之(?P<num>{_CHINESE_NUMBER_RE})", text
    ):
        add(match["den"])
        add(match["num"])

    # 百分之百 and similar expressions.
    for match in re.finditer(rf"百分(?:之)?(?P<number>{_CHINESE_NUMBER_RE})", text):
        add(match["number"])

    # Years such as 二十世纪五十年代 represent 1950; also recognize the
    # common Arabic-century form 20世纪50年代 before general digit parsing.
    for match in re.finditer(
        rf"(?P<century>{_CHINESE_NUMBER_RE})\s*世纪\s*(?P<decade>{_CHINESE_NUMBER_RE})\s*年代", text
    ):
        century = _parse_chinese_number(match["century"])
        decade = _parse_chinese_number(match["decade"])
        if century is not None and decade is not None:
            values.add(_number_key(str((century - 1) * 100 + decade)))
    for match in re.finditer(r"(?P<century>\d{1,2})\s*世纪\s*(?P<decade>\d{1,2})\s*年代", text):
        values.add(_number_key(str((int(match["century"]) - 1) * 100 + int(match["decade"]))))

    # Colloquial Chinese percentages and odds: 四成/六成 and 五五开.
    for match in re.finditer(rf"(?P<number>{_CHINESE_NUMBER_RE})成", text):
        number = _parse_chinese_number(match["number"])
        if number is not None:
            values.add(_number_key(str(number * 10)))
    for match in re.finditer(
        rf"(?P<left>{_CHINESE_NUMBER_RE})(?P<right>{_CHINESE_NUMBER_RE})开", text
    ):
        left = _parse_chinese_number(match["left"])
        right = _parse_chinese_number(match["right"])
        if left is not None and right is not None:
            values.add(_number_key(str(left * 10)))
            values.add(_number_key(str(right * 10)))
    for match in re.finditer(
        rf"(?P<left>{_CHINESE_NUMBER_RE})比(?P<right>{_CHINESE_NUMBER_RE})", text
    ):
        left = _parse_chinese_number(match["left"])
        right = _parse_chinese_number(match["right"])
        if left is not None and right is not None and left < 10 and right < 10:
            values.add(_number_key(str(left * 10)))
            values.add(_number_key(str(right * 10)))

    # Once a century has been named, Chinese commonly shortens the following
    # decades: 20世纪50年代和60年代.
    if re.search(r"20\s*世纪", text):
        for match in re.finditer(rf"(?P<decade>{_CHINESE_NUMBER_RE}|\d{{1,2}})\s*年代", text):
            decade = match["decade"]
            parsed = int(decade) if decade.isdigit() else _parse_chinese_number(decade)
            if parsed is not None:
                values.add(_number_key(str(1900 + parsed)))

    # Explicit ordinals, system labels, and common measure classifiers.
    contexts = (
        rf"第(?P<number>{_CHINESE_NUMBER_RE})",
        rf"系统(?P<number>{_CHINESE_NUMBER_RE})",
        rf"(?P<number>{_CHINESE_NUMBER_RE})(?:个|名|组|项|次|轮|章|年|月|日|小时|分钟|美元|人|种|件|倍|大)",
    )
    for pattern in contexts:
        for match in re.finditer(pattern, text):
            add(match["number"])
    return values


def _number_key(token: str) -> str:
    """Compare numeric values, not presentation-only zero/grouping changes."""
    try:
        value = format(Decimal(token.replace(",", "")), "f")
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        return value or "0"
    except InvalidOperation:
        return token
