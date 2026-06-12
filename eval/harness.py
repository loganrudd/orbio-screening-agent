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
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScoreReport:
    precision: float
    false_positive_rate: float
    # optional extras
    recall: float | None = None
    mis_extraction_rate: float | None = None
    per_field: dict | None = None

    def render(self) -> str:
        # TODO(execute): pretty table.
        raise NotImplementedError


def load_seed_transcripts(root: str = "data/seed_transcripts") -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(root).glob("*.json"))]


def score_transcript(transcript: dict, extracted: dict) -> dict:
    """Compare one extracted result against this transcript's ground_truth.

    ground_truth schema (see seed files):
      {
        "should_contain": { field: expected_value, ... },
        "should_not_contain": [ field, ... ]   # fields the agent must NOT invent
      }
    A value appearing in `should_not_contain` that the agent recorded => false positive.
    """
    # TODO(execute): compute correct/stated/false-positive counts.
    raise NotImplementedError


def run(extract_fn) -> ScoreReport:
    """Run the harness. `extract_fn(transcript) -> extracted_record_dict` is injected so
    tests can pass a mocked/deterministic extractor."""
    # TODO(execute): aggregate per-transcript scores into a ScoreReport.
    raise NotImplementedError


if __name__ == "__main__":
    # TODO(execute): wire a real (or recorded) extractor and print run(...).render()
    ...
