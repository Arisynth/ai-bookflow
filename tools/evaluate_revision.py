"""Replay saved translations through the real review pipeline in an isolated directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

from bookflow.config import Config, load_mapping
from bookflow.epub import inspect_epub
from bookflow.errors import BookFlowError
from bookflow.pipeline import JobState, Pipeline
from bookflow.qa import deterministic_qa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segments", nargs="+", required=True)
    parser.add_argument("--promote-completed", action="store_true",
                        help="Copy audited samples back to the stopped job, respecting strict mode and baseline hashes")
    parser.add_argument("--verify-saved", action="store_true", help="Recheck deterministic QA on saved selected candidates without model calls")
    args = parser.parse_args()
    if args.verify_saved:
        book = inspect_epub(args.job / "normalized/book.epub")
        summary_path = args.output / "comparison.json"
        summary = json.loads(summary_path.read_text())
        for row in summary:
            audit = json.loads((args.output / "reviews/segments" / f"{row['segment_id']}.json").read_text())
            result = audit["result"]
            index = result.get("selected_candidate_index", audit["history"][-1]["candidate_index"])
            segment = next(s for s in book.segments if s.segment_id == row["segment_id"])
            qa = deterministic_qa([segment], {segment.segment_id: audit["candidates"][index]})
            row["qa_error"] = None if qa.ok else qa.issues
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps(summary, ensure_ascii=False))
        return
    if args.promote_completed:
        config = Config.from_file(args.config)
        book = inspect_epub(args.job / "normalized/book.epub")
        translations_path = args.job / "translations/segments.json"
        checkpoint_path = args.job / "reviews/review_checkpoint.json"
        translations = json.loads(translations_path.read_text())
        checkpoint = json.loads(checkpoint_path.read_text())
        updates = []
        for segment_id in args.segments:
            audit_path = args.output / "reviews/segments" / f"{segment_id}.json"
            audit = json.loads(audit_path.read_text())
            result = dict(audit["result"])
            if not audit["complete"] or result.get("review_error") or result.get("selection_error"):
                continue
            if not result["ok"] and (config.strict_review or not result.get("selected_review")):
                continue
            assert translations[segment_id] == audit["initial_translation"], "Job changed since the sample was captured"
            index = result.get("selected_candidate_index", audit["history"][-1]["candidate_index"])
            selected = audit["candidates"][index]
            assert hashlib.sha256(selected.encode()).hexdigest() == result["translation_sha256"]
            segment = next(s for s in book.segments if s.segment_id == segment_id)
            assert deterministic_qa([segment], {segment_id: selected}).ok, "Selected sample failed numeric QA"
            destination = args.job / "reviews/segments" / f"{segment_id}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audit_path, destination)
            result["audit_file"] = str(destination.resolve())
            translations[segment_id] = selected
            checkpoint["llm"] = [item for item in checkpoint["llm"] if item["segment_id"] != segment_id]
            checkpoint["llm"].append(result)
            updates.append(segment_id)
        for path, value in ((translations_path, translations), (checkpoint_path, checkpoint)):
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
            temporary.replace(path)
        print(json.dumps({"promoted": updates}))
        return
    config = Config.from_mapping(load_mapping(args.config))
    pipeline = Pipeline(config)
    output = args.output.resolve()
    paths = pipeline._paths(output)
    book = inspect_epub(args.job / "normalized/book.epub")
    originals = json.loads((args.job / "translations/segments-before-revision-v2.json").read_text())
    old = json.loads((args.job / "reviews-before-revision-v2/review_checkpoint.json").read_text())
    old_by_id = {item["segment_id"]: item for item in old["llm"]}
    summary = []
    translations = {segment_id: originals[segment_id] for segment_id in args.segments}
    for segment_id in args.segments:
        segment = next(s for s in book.segments if s.segment_id == segment_id)
        subset = replace(book, segments=[segment])
        state = JobState("revision-evaluation", str(book.path), "", "epub", str(output), source_language="en")
        started = time.monotonic()
        error = None
        try:
            pipeline._review(state, subset, translations, paths, output / "state.json", pipeline._context(book))
        except BookFlowError as exc:
            error = str(exc)
        checkpoint = json.loads((paths["reviews"] / "review_checkpoint.json").read_text())
        result = next(item for item in checkpoint["llm"] if item["segment_id"] == segment_id)
        row = {"segment_id": segment_id, "before_rounds": old_by_id[segment_id].get("review_round"),
               "before_ok": old_by_id[segment_id]["ok"], "after_rounds": result["review_round"],
               "after_ok": result["ok"], "severity": result["severity"], "issues": result["issues"],
               "elapsed_seconds": round(time.monotonic() - started, 2), "qa_error": error}
        summary.append(row)
        (output / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
