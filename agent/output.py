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
    """Candidate-facing confirmation in natural prose.

    Each sentence ends with a period so TTS pauses naturally between fields.
    Also reads clearly as plain text. No confidence scores or internal flags.
    """

    def _val(fname: str):
        ef = getattr(record, fname)
        return ef.value if ef else None

    sentences: list[str] = []

    name = _val("candidate_name")
    position = _val("position_applied_for")
    experience = _val("years_experience")
    skills = _val("relevant_skills")
    availability = _val("availability")
    start_date = _val("earliest_start_date")
    work_auth = _val("work_authorization")
    location = _val("location_preference")

    # Name + position together
    if name and position:
        pos_str = str(position).replace("_", " ")
        sentences.append(f"I have you down as {name}, applying for the {pos_str} position.")
    elif name:
        sentences.append(f"I have you down as {name}.")
    elif position:
        sentences.append(f"You're applying for the {str(position).replace('_', ' ')} position.")

    # Experience + skills together when both present
    if experience is not None and skills:
        yrs = f"{experience} year{'s' if experience != 1 else ''}"
        skill_str = ", ".join(str(s).replace("_", " ") for s in skills) if isinstance(skills, list) else str(skills).replace("_", " ")
        sentences.append(f"You have {yrs} of experience, with skills in {skill_str}.")
    elif experience is not None:
        yrs = f"{experience} year{'s' if experience != 1 else ''}"
        sentences.append(f"You have {yrs} of experience.")
    elif skills:
        skill_str = ", ".join(str(s).replace("_", " ") for s in skills) if isinstance(skills, list) else str(skills).replace("_", " ")
        sentences.append(f"Your skills include {skill_str}.")

    # Availability
    if availability:
        avail_list = [str(a).replace("_", " ") for a in availability] if isinstance(availability, list) else [str(availability).replace("_", " ")]
        if len(avail_list) == 1:
            sentences.append(f"You're available {avail_list[0]}.")
        elif len(avail_list) == 2:
            sentences.append(f"You're available {avail_list[0]} and {avail_list[1]}.")
        else:
            avail_str = ", ".join(avail_list[:-1]) + f", and {avail_list[-1]}"
            sentences.append(f"You're available {avail_str}.")

    # Start date
    if start_date == "immediate":
        sentences.append("You can start right away.")
    elif start_date:
        sentences.append(f"Your earliest start date is {start_date}.")

    # Work authorization
    if work_auth is True:
        sentences.append("You're authorized to work.")
    elif work_auth is False:
        sentences.append("You mentioned you're not currently authorized to work.")

    # Optional location
    if location:
        sentences.append(f"Your preferred location is {location}.")

    # Call out anything we didn't capture
    missing = [
        _FIELD_LABELS.get(f, f)
        for f in ScreeningRecord.required_fields()
        if getattr(record, f) is None
    ]
    if missing:
        sentences.append(f"I wasn't able to capture: {', '.join(missing)}.")

    return " ".join(sentences)


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
