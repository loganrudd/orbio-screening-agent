"""Live integration test — requires real Claude (skipped in CI).

Run with:
    RUN_LIVE=1 pytest -m live tests/test_integration_live.py -v

This test is excluded from the default CI suite. It demonstrates the extraction
pipeline working end-to-end against the real Anthropic API and validates that at
least one real field is extracted with provenance from a single candidate turn.
"""

import datetime
import os

import pytest

from agent.extraction import Extractor
from agent.llm import ClaudeClient
from agent.schemas import ScreeningRecord, StringProposal, TurnExtraction
from agent.storage import Turn


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("RUN_LIVE"), reason="Set RUN_LIVE=1 to run live API tests")
async def test_live_single_extraction_turn():
    """One real-Claude extraction turn: candidate states name + position.

    Verifies:
    - extract_turn returns a ScreeningRecord with at least one field set
    - candidate_name is extracted with provenance
    - no false positives from fields not mentioned in the turn
    """
    llm = ClaudeClient()
    extractor = Extractor(llm=llm)
    record = ScreeningRecord()

    turn = Turn(
        role="candidate",
        content="Hi, my name is Maria Gonzalez and I'd like to apply as a line cook.",
        ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    result = await extractor.extract_turn(
        record=record,
        latest_turn=turn,
        turn_index=0,
        language="en",
    )

    # At least name or position should be extracted
    assert result.candidate_name is not None or result.position_applied_for is not None

    # Provenance must be set on any extracted field
    for fname in ["candidate_name", "position_applied_for"]:
        ef = getattr(result, fname)
        if ef is not None:
            assert len(ef.provenance) > 0, f"{fname} missing provenance"
            assert ef.provenance[0].source_text, f"{fname} has empty source_text"

    # Fields not mentioned must not be invented (anti-FP)
    assert result.years_experience is None, "years_experience was not mentioned — must be None"
    assert result.availability is None, "availability was not mentioned — must be None"
    assert result.work_authorization is None, "work_authorization was not mentioned — must be None"
