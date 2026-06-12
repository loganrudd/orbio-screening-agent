"""Deterministic eval harness: extraction precision + false-positive rate.

Replays labeled seed transcripts through the extraction logic and scores results
against ground truth. DETERMINISTIC scoring (no LLM judge for the core metrics). Must
run without network/credentials (mock the LLM or use pre-recorded extraction outputs).
See eval.md.

Definitions (load-bearing — see CLAUDE.md):
  false positive  = field recorded that the candidate never stated
  precision       = correct-AND-stated / claimed-collected
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class FieldScore:
    correct_stated: int = 0
    mis_extraction: int = 0
    false_negative: int = 0
    false_positive: int = 0
    conflict_correct: int = 0  # correctly surfaced CONFLICTING fields (not in precision denominator)


@dataclass
class ScoreReport:
    precision: float
    false_positive_rate: float
    recall: float | None = None
    mis_extraction_rate: float | None = None
    per_field: dict[str, FieldScore] | None = None

    def render(self) -> str:
        lines: list[str] = [
            "",
            "═" * 60,
            "  EXTRACTION EVAL REPORT",
            "═" * 60,
            f"  Precision:          {self.precision:.3f}",
            f"  False-Positive Rate:{self.false_positive_rate:.3f}",
        ]
        if self.recall is not None:
            lines.append(f"  Recall:             {self.recall:.3f}")
        if self.mis_extraction_rate is not None:
            lines.append(f"  Mis-extraction Rate:{self.mis_extraction_rate:.3f}")

        if self.per_field:
            lines.append("─" * 60)
            lines.append(
                "  {:<28} {:>6} {:>6} {:>6} {:>6}".format(
                    "Field", "TP", "FP", "FN", "Mis"
                )
            )
            lines.append("─" * 60)
            for fname, fs in sorted(self.per_field.items()):
                lines.append(
                    "  {:<28} {:>6} {:>6} {:>6} {:>6}".format(
                        fname,
                        fs.correct_stated,
                        fs.false_positive,
                        fs.false_negative,
                        fs.mis_extraction,
                    )
                )
        lines.extend(["═" * 60, ""])
        return "\n".join(lines)


def load_seed_transcripts(root: str = "data/seed_transcripts") -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(root).glob("*.json"))]


# --------------------------------------------------------------------------- scoring


def _normalize(value: Any) -> Any:
    """Normalize a value for comparison: lower-case strings, sorted str lists."""
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, list):
        return sorted(str(v).strip().lower() for v in value)
    return value


def _values_match(extracted: Any, expected: Any) -> bool:
    """True when the extracted value matches the expected ground-truth value."""
    return _normalize(extracted) == _normalize(expected)


def _field_is_recorded(extracted_record: dict, field_name: str) -> bool:
    """True when the extracted record has a non-None value for this field."""
    ef = extracted_record.get(field_name)
    if ef is None:
        return False
    if isinstance(ef, dict):
        return ef.get("value") is not None
    return True


def _field_value(extracted_record: dict, field_name: str) -> Any:
    """Extract the canonical value from the record dict."""
    ef = extracted_record.get(field_name)
    if ef is None:
        return None
    if isinstance(ef, dict):
        return ef.get("value")
    return ef


def _field_flag(extracted_record: dict, field_name: str) -> str | None:
    """Return the flag string for a field, or None if field absent."""
    ef = extracted_record.get(field_name)
    if isinstance(ef, dict):
        return ef.get("flag")
    return None


def _conflicting_values(extracted_record: dict, field_name: str) -> list[Any]:
    ef = extracted_record.get(field_name)
    if isinstance(ef, dict):
        return ef.get("conflicting_values", [])
    return []


def score_transcript(transcript: dict, extracted: dict) -> dict:
    """Compare one extracted result against this transcript's ground_truth.

    ground_truth schema (see seed files):
      {
        "should_contain": { field: expected_value, ... },
        "should_not_contain": [ field, ... ]   # fields the agent must NOT invent
        "conflicting": { field: [value1, value2, ...], ... }  # optional
      }
    A value appearing in `should_not_contain` that the agent recorded => false positive.
    A conflicting field handled correctly (flag=CONFLICTING, values cover expected set)
    is scored as correct-handling — excluded from the precision denominator.

    Returns a counts dict:
      {
        "claimed_collected": int,     # recorded, non-conflicting fields
        "correct_stated": int,        # claimed and correct
        "mis_extraction": int,        # claimed but wrong value
        "false_negative": int,        # should_contain but not recorded
        "false_positive": int,        # should_not_contain but recorded
        "conflict_correct": int,      # conflicting fields correctly surfaced
        "conflict_missed": int,       # conflicting fields NOT surfaced
        "per_field": dict[str, FieldScore],
      }
    """
    ground_truth = transcript.get("ground_truth", {})
    should_contain: dict[str, Any] = ground_truth.get("should_contain", {})
    should_not_contain: list[str] = ground_truth.get("should_not_contain", [])
    conflicting: dict[str, list[Any]] = ground_truth.get("conflicting", {})

    claimed_collected = 0
    correct_stated = 0
    mis_extraction = 0
    false_negative = 0
    false_positive = 0
    conflict_correct = 0
    conflict_missed = 0
    per_field: dict[str, FieldScore] = {}

    # Score should_contain fields
    for field_name, expected_value in should_contain.items():
        fs = per_field.setdefault(field_name, FieldScore())
        recorded = _field_is_recorded(extracted, field_name)
        if recorded:
            flag = _field_flag(extracted, field_name)
            if flag == "conflicting":
                # Treat a conflicting flag on a should_contain field as mis-extraction
                # (agent should have confirmed a single value — not split the issue)
                mis_extraction += 1
                claimed_collected += 1
                fs.mis_extraction += 1
            else:
                claimed_collected += 1
                extracted_value = _field_value(extracted, field_name)
                if _values_match(extracted_value, expected_value):
                    correct_stated += 1
                    fs.correct_stated += 1
                else:
                    mis_extraction += 1
                    fs.mis_extraction += 1
        else:
            false_negative += 1
            fs.false_negative += 1

    # Score should_not_contain fields
    for field_name in should_not_contain:
        fs = per_field.setdefault(field_name, FieldScore())
        if _field_is_recorded(extracted, field_name):
            false_positive += 1
            claimed_collected += 1
            fs.false_positive += 1

    # Score conflicting fields (separate sub-metric; NOT in precision denominator)
    for field_name, expected_values in conflicting.items():
        fs = per_field.setdefault(field_name, FieldScore())
        flag = _field_flag(extracted, field_name)
        if flag == "conflicting":
            # Check that the union of value + conflicting_values covers expected set
            main_val = _field_value(extracted, field_name)
            extra_vals = _conflicting_values(extracted, field_name)
            recorded_set = {_normalize(v) for v in ([main_val] + extra_vals) if v is not None}
            expected_set = {_normalize(v) for v in expected_values}
            if expected_set.issubset(recorded_set):
                conflict_correct += 1
                fs.conflict_correct += 1
            else:
                conflict_missed += 1
        else:
            conflict_missed += 1

    return {
        "claimed_collected": claimed_collected,
        "correct_stated": correct_stated,
        "mis_extraction": mis_extraction,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "conflict_correct": conflict_correct,
        "conflict_missed": conflict_missed,
        "per_field": per_field,
    }


def run(extract_fn: Callable[[dict], dict]) -> ScoreReport:
    """Run the harness over all seed transcripts.

    `extract_fn(transcript) -> extracted_record_dict` is injected so tests and the
    offline replay path can pass a mocked/deterministic extractor.
    """
    transcripts = load_seed_transcripts()
    if not transcripts:
        raise RuntimeError("No seed transcripts found in data/seed_transcripts/")

    totals = {
        "claimed_collected": 0,
        "correct_stated": 0,
        "mis_extraction": 0,
        "false_negative": 0,
        "false_positive": 0,
    }
    per_field_agg: dict[str, FieldScore] = {}

    for transcript in transcripts:
        # Skip non-evaluation seed files (e.g. files in recorded/ subdirectory)
        if "ground_truth" not in transcript:
            continue
        extracted = extract_fn(transcript)
        scores = score_transcript(transcript, extracted)

        for key in totals:
            totals[key] += scores[key]
        for fname, fs in scores["per_field"].items():
            agg = per_field_agg.setdefault(fname, FieldScore())
            agg.correct_stated += fs.correct_stated
            agg.mis_extraction += fs.mis_extraction
            agg.false_negative += fs.false_negative
            agg.false_positive += fs.false_positive
            agg.conflict_correct += fs.conflict_correct

    claimed = totals["claimed_collected"]
    correct = totals["correct_stated"]
    fp = totals["false_positive"]
    fn = totals["false_negative"]
    mis = totals["mis_extraction"]

    precision = correct / claimed if claimed > 0 else 0.0
    fp_rate = fp / claimed if claimed > 0 else 0.0
    recall_denom = correct + fn
    recall = correct / recall_denom if recall_denom > 0 else None
    mis_rate = mis / claimed if claimed > 0 else None

    return ScoreReport(
        precision=precision,
        false_positive_rate=fp_rate,
        recall=recall,
        mis_extraction_rate=mis_rate,
        per_field=per_field_agg,
    )


if __name__ == "__main__":
    import argparse

    from eval.replay import replay_extract_fn

    parser = argparse.ArgumentParser(description="Run the extraction eval harness (offline replay).")
    parser.add_argument(
        "--recorded-dir",
        default="data/seed_transcripts/recorded",
        help="Directory containing recorded proposal fixtures (default: data/seed_transcripts/recorded)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use a real ClaudeClient instead of the offline replay extractor (requires ANTHROPIC_API_KEY).",
    )
    args = parser.parse_args()

    if args.live:
        import asyncio
        import datetime as _dt
        from agent.extraction import Extractor as _Extractor
        from agent.llm import ClaudeClient
        from agent.schemas import ScreeningRecord as _SR
        from agent.storage import Turn as _Turn

        def _live_extract(transcript: dict) -> dict:
            llm = ClaudeClient()
            extractor = _Extractor(llm=llm)
            record = _SR()
            lang = transcript.get("language", "en")
            candidate_turns = [t for t in transcript.get("turns", []) if t["role"] == "candidate"]

            async def _run() -> _SR:
                nonlocal record
                for idx, raw in enumerate(candidate_turns):
                    turn = _Turn(
                        role="candidate",
                        content=raw["content"],
                        ts=_dt.datetime.now(_dt.timezone.utc).isoformat(),
                    )
                    record = await extractor.extract_turn(
                        record=record, latest_turn=turn, turn_index=idx, language=lang
                    )
                return record

            return asyncio.run(_run()).model_dump(mode="json")

        extract_fn = _live_extract
    else:
        extract_fn = replay_extract_fn(args.recorded_dir)

    report = run(extract_fn)
    print(report.render())
