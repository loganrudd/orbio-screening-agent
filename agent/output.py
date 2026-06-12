"""Reviewer-facing output: confidence computation, flag assignment, rendering, summary.

The output is built for a human reviewer deciding whether to advance a candidate.
See output-contract.md. Confidence is rule-derived (extraction.md), never model
self-reported. CONFIRMED threshold: score >= 0.8.
"""

from __future__ import annotations

from .schemas import Confidence, ExtractedField, FieldFlag, ScreeningRecord

# Threshold above which a field is CONFIRMED (validated + explicitly stated => 0.9)
_CONFIRMED_THRESHOLD = 0.8

# Human-readable labels for the reviewer table
_FIELD_LABELS: dict[str, str] = {
    "candidate_name": "Name",
    "position_applied_for": "Position",
    "years_experience": "Experience",
    "relevant_skills": "Skills",
    "availability": "Availability",
    "earliest_start_date": "Start Date",
    "work_authorization": "Work Auth",
    "location_preference": "Location Pref",
}

_FLAG_ICONS: dict[FieldFlag, str] = {
    FieldFlag.CONFIRMED: "✓",
    FieldFlag.NEEDS_REVIEW: "⚠",
    FieldFlag.MISSING: "✗",
    FieldFlag.CONFLICTING: "!",
}


def compute_confidence(
    *,
    validated: bool,
    explicitly_stated: bool,
    stt_confidence: float | None = None,
) -> Confidence:
    """Build a rule-derived Confidence instance (score auto-computed by the model)."""
    return Confidence(
        validated=validated,
        explicitly_stated=explicitly_stated,
        stt_confidence=stt_confidence,
    )


def assign_flag(field: ExtractedField, *, reprompt_capped: bool) -> FieldFlag:
    """Map a field's state to a reviewer flag.

    Priority: CONFLICTING > NEEDS_REVIEW (reprompt cap) > CONFIRMED/NEEDS_REVIEW by score.
    """
    if field.conflicting_values:
        return FieldFlag.CONFLICTING
    if reprompt_capped:
        return FieldFlag.NEEDS_REVIEW
    if field.confidence.score >= _CONFIRMED_THRESHOLD:
        return FieldFlag.CONFIRMED
    return FieldFlag.NEEDS_REVIEW


def render_candidate_confirmation(record: ScreeningRecord) -> str:
    """Candidate-facing field summary: values only, no confidence or flags.

    Used in the CONFIRMING state to let the candidate verify their answers
    without exposing internal reviewer metadata.
    """
    lines: list[str] = []
    all_fields = ScreeningRecord.required_fields() + ["location_preference"]
    for fname in all_fields:
        ef: ExtractedField | None = getattr(record, fname)
        label = _FIELD_LABELS.get(fname, fname)
        optional = " (optional)" if fname == "location_preference" else ""
        if ef is None:
            lines.append(f"  {label}{optional}: (not provided)")
        else:
            lines.append(f"  {label}: {_format_value(ef.value)}")
    return "\n".join(lines)


def render_reviewer_table(record: ScreeningRecord) -> str:
    """Render a formatted terminal table with per-field confidence + flags.

    ✓ confirmed / ⚠ needs_review / ✗ missing / ! conflicting
    """
    col_w = (24, 32, 5)  # field, value, conf
    divider = "─" * (sum(col_w) + 14)
    header = "═" * (sum(col_w) + 14)

    lines: list[str] = [
        "",
        header,
        "  CANDIDATE SCREENING REVIEW",
        header,
        "  {:<{}} {:<{}} {:>{}}  {}".format(
            "Field", col_w[0], "Value", col_w[1], "Conf", col_w[2], "Status"
        ),
        divider,
    ]

    all_fields = ScreeningRecord.required_fields() + ["location_preference"]
    for fname in all_fields:
        ef: ExtractedField | None = getattr(record, fname)
        label = _FIELD_LABELS.get(fname, fname)
        optional_note = " (opt)" if fname == "location_preference" else ""

        if ef is None:
            icon = _FLAG_ICONS[FieldFlag.MISSING]
            val_str = "(not provided)"
            conf_str = " — "
            flag_str = FieldFlag.MISSING.value
        else:
            icon = _FLAG_ICONS.get(ef.flag, "?")
            val_str = _format_value(ef.value)
            conf_str = f"{ef.confidence.score:.2f}"
            flag_str = ef.flag.value

        # Truncate long values
        if len(val_str) > col_w[1]:
            val_str = val_str[: col_w[1] - 1] + "…"

        lines.append(
            "  {} {:<{}} {:<{}} {:>{}}  {}{}".format(
                icon,
                label + optional_note,
                col_w[0],
                val_str,
                col_w[1],
                conf_str,
                col_w[2],
                flag_str,
                "",
            )
        )

    lines.extend([divider, header, ""])
    return "\n".join(lines)


def build_summary(record: ScreeningRecord) -> str:
    """Structured synopsis for the reviewer: what to trust, what to verify."""
    lines: list[str] = ["SCREENING SUMMARY", "─" * 40]

    def _val(fname: str) -> str:
        ef = getattr(record, fname)
        return str(ef.value) if ef else "(missing)"

    # Narrative line
    name = _val("candidate_name")
    position = _val("position_applied_for").replace("_", " ")
    exp = _val("years_experience")
    lines.append(f"Candidate:  {name}")
    lines.append(f"Position:   {position}")
    lines.append(f"Experience: {exp} year(s)")

    skills_ef = record.relevant_skills
    if skills_ef:
        skills_str = ", ".join(str(s) for s in skills_ef.value) if isinstance(skills_ef.value, list) else str(skills_ef.value)
        lines.append(f"Skills:     {skills_str}")

    avail_ef = record.availability
    if avail_ef:
        avail_str = ", ".join(str(s).replace("_", " ") for s in avail_ef.value) if isinstance(avail_ef.value, list) else str(avail_ef.value)
        lines.append(f"Available:  {avail_str}")

    lines.append(f"Start Date: {_val('earliest_start_date')}")
    lines.append(f"Work Auth:  {'Yes' if record.work_authorization and record.work_authorization.value else 'No/Missing'}")

    loc_ef = record.location_preference
    if loc_ef:
        lines.append(f"Location:   {loc_ef.value}")

    lines.append("")

    # Confidence / flag summary
    required = ScreeningRecord.required_fields()
    confirmed = [f for f in required if getattr(record, f) and getattr(record, f).flag == FieldFlag.CONFIRMED]
    needs_review = [f for f in required if getattr(record, f) and getattr(record, f).flag in (FieldFlag.NEEDS_REVIEW, FieldFlag.CONFLICTING)]
    missing = [f for f in required if not getattr(record, f)]

    lines.append(f"Confirmed ({len(confirmed)}/{len(required)}): {', '.join(confirmed) or 'none'}")
    if needs_review:
        lines.append(f"Needs review: {', '.join(needs_review)}")
    if missing:
        lines.append(f"Missing: {', '.join(missing)}")

    return "\n".join(lines)


def _format_value(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(v).replace("_", " ") for v in value)
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value).replace("_", " ")
