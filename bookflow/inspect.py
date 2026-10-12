from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .errors import BookFlowError


@dataclass(frozen=True)
class InputInfo:
    path: Path
    kind: str
    sha256: str
    size: int


def inspect_input(path: Path) -> InputInfo:
    path = path.expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise BookFlowError(f"输入文件不存在或不可读: {path}")
    suffix = path.suffix.lower()
    with path.open("rb") as handle:
        head = handle.read(8)
        digest = hashlib.sha256()
        digest.update(head)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if head.startswith(b"%PDF") or suffix == ".pdf":
        kind = "pdf"
    elif suffix == ".epub" or head.startswith(b"PK"):
        kind = "epub"
    elif suffix in {".mobi", ".azw", ".azw3"}:
        kind = suffix[1:]
    else:
        raise BookFlowError("不支持的输入格式；支持 PDF、EPUB、MOBI、AZW3")
    return InputInfo(path, kind, digest.hexdigest(), path.stat().st_size)
