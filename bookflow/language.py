from __future__ import annotations

import re


def detect_language(text: str) -> str:
    sample = text[:20000]
    if not sample.strip():
        return "unknown"
    chinese = len(re.findall(r"[\u4e00-\u9fff]", sample))
    latin = len(re.findall(r"[A-Za-z]", sample))
    german = len(re.findall(r"\b(der|die|das|und|nicht|ist|ein|eine|ich|für)\b", sample.lower()))
    if chinese >= max(10, latin // 3):
        return "zh-CN"
    if latin == 0:
        return "unknown"
    if german >= 2:
        return "de"
    return "en"
