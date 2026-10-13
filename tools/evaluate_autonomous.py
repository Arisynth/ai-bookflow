"""Small real-local-model regression; does not touch existing book jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bookflow.config import Config
from bookflow.epub import markdown_to_epub
from bookflow.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5:latest")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true", help="Run a short end-to-end sample")
    parser.add_argument("--automatic-only", action="store_true", help="Leave glossary and style entirely to the local model")
    args = parser.parse_args()
    if args.output is None:
        suffix = ("-quick" if args.quick else "") + ("-automatic" if args.automatic_only else "")
        args.output = Path(f"artifacts/evaluation/autonomous{suffix}-v1")
    args.output.mkdir(parents=True, exist_ok=True)
    markdown = args.output / "source.md"
    sample = (
        "# Forecasting and evidence\n\n"
        "In this referendum, the yes campaign supported separation and the no campaign opposed it. "
        "Several forecasters expected a close result.\n\n"
        "As it turned out, no won by 55.3% to 44.7%. The forecasters then compared their predictions with the outcome.\n\n"
        "## Measuring accuracy\n\n"
        "A Brier score measures the accuracy of a probabilistic forecast. Lower scores indicate more accurate forecasts. "
        "Calibration and resolution describe different properties of forecasting performance.\n\n"
        "The same Brier score must be named consistently throughout this book. "
        "One project cost 100,000 pounds; another cost 1.3 billion dollars.\n"
    )
    if args.quick:
        sample = ("# Forecasting\n\nA Brier score measures the accuracy of a probabilistic forecast. "
                  "Lower scores indicate more accurate forecasts.\n")
    source_changed = not markdown.exists() or markdown.read_text(encoding="utf-8") != sample
    markdown.write_text(sample, encoding="utf-8")
    source = args.output / "source.epub"
    if source_changed or not source.exists():
        markdown_to_epub(markdown, source, "Forecasting" if args.quick else "Forecasting and evidence")
    config = Config(workspace=args.output / "jobs", translation_model=args.model, review_model=args.model,
                    num_ctx=8192, review_num_ctx=16384, review_num_predict=4096, think=False,
                    retry_backoff_seconds=1, glossary={} if args.automatic_only else {"Brier score": "布里尔评分"},
                    style={} if args.automatic_only else {"tone": "准确、自然的中文科普说明；不要添加原文没有的解释。"})
    state = Pipeline(config).start(source)
    result = {"job_id": state.job_id, "automatic_only": args.automatic_only, "quick": args.quick,
              "quality_status": state.quality_status,
              "summary": state.quality_summary, "outputs": state.outputs}
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
