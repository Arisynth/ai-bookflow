from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from . import __version__
from .config import Config
from .errors import BookFlowError
from .pipeline import Pipeline, load_state


def _config(args: argparse.Namespace) -> Config:
    config = Config.from_file(Path(args.config).expanduser()) if getattr(args, "config", None) else Config()
    overrides = {}
    if getattr(args, "workspace", None):
        overrides["workspace"] = Path(args.workspace).expanduser()
    if getattr(args, "model", None):
        overrides["translation_model"] = args.model
    if getattr(args, "review_model", None):
        overrides["review_model"] = args.review_model
    if getattr(args, "source", None):
        overrides["source_language"] = args.source
    if getattr(args, "target", None):
        overrides["target_language"] = args.target
    if getattr(args, "mode", None):
        overrides["output_mode"] = args.mode
    if getattr(args, "no_review", False):
        overrides["review_enabled"] = False
    if getattr(args, "no_validate", False):
        overrides["validate_epub"] = False
    if overrides:
        config = replace(config, **overrides)
    return config


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="配置文件路径（YAML/JSON）")
    parser.add_argument("--workspace", help="任务工作目录，默认 ~/.bookflow")
    parser.add_argument("--model", help="覆盖翻译模型名")
    parser.add_argument("--review-model", help="覆盖审校模型名")
    parser.add_argument("--source", help="手动指定源语言，例如 en/de/zh-CN")
    parser.add_argument("--target", help="覆盖目标语言，默认 zh-CN")
    parser.add_argument("--mode", choices=["original", "translated", "bilingual"], help="输出模式")
    parser.add_argument("--no-review", action="store_true", help="关闭审校阶段")
    parser.add_argument("--no-validate", action="store_true", help="关闭内置 EPUB 校验")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bookflow", description="本地 AI 电子书处理与翻译工具")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="处理 PDF/EPUB/MOBI/AZW3")
    convert.add_argument("input", type=Path)
    convert.add_argument("--output", type=Path, help="输出目录")
    _add_common(convert)
    resume = commands.add_parser("resume", help="恢复失败或中断的任务")
    resume.add_argument("job_id")
    resume.add_argument("--output", type=Path)
    _add_common(resume)
    status = commands.add_parser("status", help="查看任务状态")
    status.add_argument("job_id")
    status.add_argument("--config")
    status.add_argument("--workspace")
    failures = commands.add_parser("failures", help="查看质量检查与任务错误")
    failures.add_argument("job_id")
    failures.add_argument("--config")
    failures.add_argument("--workspace")
    doctor = commands.add_parser("doctor", help="检查本地工具是否可用")
    doctor.add_argument("--config")
    doctor.add_argument("--workspace")
    return parser


def _print_state(state: object) -> None:
    print(json.dumps(state.__dict__, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config(args)
        if args.command == "convert":
            state = Pipeline(config).start(args.input, output_dir=args.output)
            _print_state(state)
        elif args.command == "resume":
            state = Pipeline(config).resume(args.job_id, output_dir=args.output)
            _print_state(state)
        elif args.command == "status":
            _print_state(load_state(config, args.job_id))
        elif args.command == "failures":
            state = load_state(config, args.job_id)
            _print_state(state)
            quality = Path(state.workdir) / "reviews" / "quality.json"
            if quality.exists():
                print(quality.read_text(encoding="utf-8"))
        elif args.command == "doctor":
            checks = {name: shutil.which(value) or False for name, value in {
                "ollama": "ollama", "marker": config.marker_command, "calibre": config.calibre_command, "babelebook": config.babelebook_command
            }.items()}
            print(json.dumps({"platform": sys.platform, "checks": checks}, ensure_ascii=False, indent=2))
        return 0
    except (BookFlowError, ValueError, OSError) as exc:
        print(f"BookFlow: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
