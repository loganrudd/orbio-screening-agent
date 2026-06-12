"""Tests for the eval harness scoring logic.

Uses synthetic extracted dicts — no recorded fixtures or network required.
After eval/record.py captures fixtures, an additional test verifies the
offline replay matches the expected aggregate.
"""

import pytest

from eval.harness import ScoreReport, _values_match, run, score_transcript


# ─────────────────────────── _values_match (normalizer) ──────────────────────

class TestValuesMatch:
    def test_exact_string_match(self):
        assert _values_match("server", "server") is True

    def test_case_insensitive_string(self):
        assert _values_match("Server", "server") is True

    def test_whitespace_stripped(self):
        assert _values_match("  server  ", "server") is True

    def test_string_mismatch(self):
        assert _values_match("server", "host") is False

    def test_int_match(self):
        assert _values_match(4, 4) is True

    def test_int_mismatch(self):
        assert _values_match(4, 5) is False

    def test_bool_true_match(self):
        assert _values_match(True, True) is True

    def test_bool_mismatch(self):
        assert _values_match(True, False) is False

    def test_list_exact_match(self):
        assert _values_match(["weekday_evening", "weekend_day"], ["weekday_evening", "weekend_day"]) is True

    def test_list_order_agnostic(self):
        assert _values_match(["weekend_day", "weekday_evening"], ["weekday_evening", "weekend_day"]) is True

    def test_list_case_insensitive(self):
        assert _values_match(["WEEKDAY_EVENING"], ["weekday_evening"]) is True

    def test_list_mismatch(self):
        assert _values_match(["weekday_day"], ["weekday_evening"]) is False


# ───────────────────────────── score_transcript ───────────────────────────────

def _make_extracted(**fields) -> dict:
    """Build a minimal extracted record dict from kwargs of the form field_name=value."""
    result = {}
    for name, val in fields.items():
        if val is None:
            result[name] = None
        elif isinstance(val, dict):
            result[name] = val
        else:
            result[name] = {"value": val, "flag": "confirmed", "conflicting_values": []}
    return result


def _conflicting_field(main_val, alt_val) -> dict:
    return {"value": main_val, "flag": "conflicting", "conflicting_values": [alt_val]}


class TestScoreTranscript:
    def _transcript(self, should_contain=None, should_not_contain=None, conflicting=None):
        gt = {}
        if should_contain:
            gt["should_contain"] = should_contain
        if should_not_contain:
            gt["should_not_contain"] = should_not_contain
        if conflicting:
            gt["conflicting"] = conflicting
        return {"id": "test", "ground_truth": gt}

    def test_clean_perfect_extraction_precision_1(self):
        t = self._transcript(
            should_contain={"candidate_name": "Maria", "years_experience": 4}
        )
        extracted = _make_extracted(candidate_name="Maria", years_experience=4)
        s = score_transcript(t, extracted)
        assert s["correct_stated"] == 2
        assert s["false_positive"] == 0
        assert s["false_negative"] == 0
        assert s["claimed_collected"] == 2

    def test_false_positive_detected(self):
        """Field in should_not_contain that IS recorded → false positive."""
        t = self._transcript(
            should_contain={"candidate_name": "James"},
            should_not_contain=["years_experience"],
        )
        extracted = _make_extracted(
            candidate_name="James",
            years_experience=5,  # candidate never stated this
        )
        s = score_transcript(t, extracted)
        assert s["false_positive"] == 1
        assert s["correct_stated"] == 1

    def test_false_negative_detected(self):
        """should_contain field NOT in extracted → false negative."""
        t = self._transcript(should_contain={"candidate_name": "Maria", "years_experience": 4})
        extracted = _make_extracted(candidate_name="Maria")  # years_experience missing
        s = score_transcript(t, extracted)
        assert s["false_negative"] == 1
        assert s["correct_stated"] == 1

    def test_mis_extraction_detected(self):
        """Field recorded but wrong value → mis-extraction, not FP."""
        t = self._transcript(should_contain={"years_experience": 4})
        extracted = _make_extracted(years_experience=3)  # wrong value
        s = score_transcript(t, extracted)
        assert s["mis_extraction"] == 1
        assert s["correct_stated"] == 0
        assert s["false_positive"] == 0  # it's in should_contain, not FP

    def test_should_not_contain_field_absent_is_ok(self):
        """Field in should_not_contain that is NOT recorded → correct (no FP)."""
        t = self._transcript(
            should_contain={"candidate_name": "James"},
            should_not_contain=["years_experience"],
        )
        extracted = _make_extracted(candidate_name="James")  # years_experience absent — correct
        s = score_transcript(t, extracted)
        assert s["false_positive"] == 0

    def test_conflicting_field_correctly_surfaced(self):
        """Conflicting field with both values present → conflict_correct, not in precision denom."""
        t = self._transcript(
            conflicting={"position_applied_for": ["server", "host"]}
        )
        extracted = _make_extracted(
            position_applied_for=_conflicting_field("server", "host")
        )
        s = score_transcript(t, extracted)
        assert s["conflict_correct"] == 1
        assert s["claimed_collected"] == 0  # conflicting not in precision denominator
        assert s["false_positive"] == 0

    def test_conflicting_field_not_surfaced_is_conflict_missed(self):
        """Conflicting field recorded without CONFLICTING flag → conflict_missed."""
        t = self._transcript(
            conflicting={"position_applied_for": ["server", "host"]}
        )
        extracted = _make_extracted(position_applied_for="server")  # no conflict flag
        s = score_transcript(t, extracted)
        assert s["conflict_missed"] == 1
        assert s["conflict_correct"] == 0

    def test_per_field_breakdown(self):
        t = self._transcript(
            should_contain={"candidate_name": "Maria"},
            should_not_contain=["years_experience"],
        )
        extracted = _make_extracted(
            candidate_name="Maria",
            years_experience=5,
        )
        s = score_transcript(t, extracted)
        assert "candidate_name" in s["per_field"]
        assert s["per_field"]["candidate_name"].correct_stated == 1
        assert "years_experience" in s["per_field"]
        assert s["per_field"]["years_experience"].false_positive == 1

    def test_list_value_order_agnostic(self):
        t = self._transcript(
            should_contain={"availability": ["weekday_evening", "weekend_day"]}
        )
        extracted = _make_extracted(availability=["weekend_day", "weekday_evening"])  # reversed
        s = score_transcript(t, extracted)
        assert s["correct_stated"] == 1


# ─────────────────────────────────── run() ────────────────────────────────────

class TestRun:
    def _make_extract_fn(self, result: dict):
        """Always returns the same extracted dict regardless of transcript."""
        def extract_fn(transcript: dict) -> dict:
            return result
        return extract_fn

    def test_run_returns_score_report(self, tmp_path, monkeypatch):
        """run() over a single synthetic transcript yields a valid ScoreReport."""
        # Patch load_seed_transcripts to return one clean transcript
        fake_transcript = {
            "id": "fake",
            "ground_truth": {
                "should_contain": {"candidate_name": "Maria"},
                "should_not_contain": [],
            }
        }
        monkeypatch.setattr("eval.harness.load_seed_transcripts", lambda: [fake_transcript])

        extracted = _make_extracted(candidate_name="Maria")
        report = run(self._make_extract_fn(extracted))

        assert isinstance(report, ScoreReport)
        assert report.precision == pytest.approx(1.0)
        assert report.false_positive_rate == pytest.approx(0.0)

    def test_run_detects_false_positive(self, monkeypatch):
        fake_transcript = {
            "id": "fake",
            "ground_truth": {
                "should_contain": {},
                "should_not_contain": ["years_experience"],
            }
        }
        monkeypatch.setattr("eval.harness.load_seed_transcripts", lambda: [fake_transcript])

        extracted = _make_extracted(years_experience=5)
        report = run(self._make_extract_fn(extracted))

        assert report.false_positive_rate == pytest.approx(1.0)
        assert report.precision == pytest.approx(0.0)

    def test_score_report_render_contains_precision(self, monkeypatch):
        fake_transcript = {
            "id": "fake",
            "ground_truth": {"should_contain": {"candidate_name": "Maria"}, "should_not_contain": []}
        }
        monkeypatch.setattr("eval.harness.load_seed_transcripts", lambda: [fake_transcript])

        extracted = _make_extracted(candidate_name="Maria")
        report = run(self._make_extract_fn(extracted))
        rendered = report.render()

        assert "Precision" in rendered
        assert "False-Positive" in rendered
