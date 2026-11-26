"""F-07 release gate. Thresholds are data; failed outputs stay in denominators.

Rules implemented from the issue:

* pooled human syllable error <= 5% and <= 10% per declared bucket;
* targeted contrast accuracy >= 95% with no unresolved systematic critical error;
* naturalness mean >= 4.0 and listener/utterance-aware 95% lower bound >= 3.8,
  with a gap from matched natural speech <= 0.5;
* no catastrophic loops/truncation/language switch in the suite;
* no missing/repeated clauses in the paragraphs;
* acceptance by Jee and the native evaluation panel.

Insufficient evidence yields ``inconclusive``; it never becomes a pass.
"""

from __future__ import annotations

from typing import Any

from ..errors import InputError

VERDICTS = ("accepted", "rejected", "inconclusive")


def evaluate_release(
    *,
    thresholds: dict[str, float],
    syllable: dict[str, Any],
    contrast: dict[str, Any],
    naturalness: dict[str, Any],
    natural: dict[str, Any],
    paragraphs: dict[str, Any],
    catastrophic_outputs: int,
    panel_acceptance: bool | None,
    required_buckets: list[str] | None = None,
) -> dict[str, Any]:
    """Return accepted / rejected / inconclusive with typed reasons."""
    reasons: list[dict[str, str]] = []
    rejects: list[str] = []
    inconclusive: list[str] = []

    required = {
        "pooled_syllable_error_max",
        "bucket_syllable_error_max",
        "contrast_accuracy_min",
        "naturalness_mean_min",
        "naturalness_lower_bound_min",
        "naturalness_gap_max",
    }
    missing_thresholds = required - set(thresholds)
    if missing_thresholds:
        raise InputError(f"release thresholds are missing: {sorted(missing_thresholds)}")

    total = syllable.get("totals", {})
    reference_syllables = int(total.get("reference_syllables", 0))
    if reference_syllables == 0:
        inconclusive.append("syllable_instrument_no_data")
        reasons.append(
            {
                "code": "no_transcription_data",
                "kind": "inconclusive",
                "detail": "no pooled reference syllables were scored",
            }
        )
    else:
        pooled = float(total.get("error_rate", 0.0))
        if pooled > float(thresholds["pooled_syllable_error_max"]):
            rejects.append("pooled_syllable_error")
            reasons.append(
                {
                    "code": "pooled_syllable_error",
                    "kind": "rejected",
                    "detail": (
                        f"pooled error {pooled:.4f} exceeds "
                        f"{thresholds['pooled_syllable_error_max']}"
                    ),
                }
            )
        buckets = syllable.get("by_bucket", {})
        check_buckets = required_buckets or sorted(buckets)
        for bucket in check_buckets:
            if bucket not in buckets or int(buckets[bucket].get("reference_syllables", 0)) == 0:
                inconclusive.append(f"bucket_no_data:{bucket}")
                reasons.append(
                    {
                        "code": "bucket_no_data",
                        "kind": "inconclusive",
                        "detail": f"bucket {bucket!r} has no scored items",
                    }
                )
                continue
            rate = float(buckets[bucket].get("error_rate", 0.0))
            if rate > float(thresholds["bucket_syllable_error_max"]):
                rejects.append(f"bucket_syllable_error:{bucket}")
                reasons.append(
                    {
                        "code": "bucket_syllable_error",
                        "kind": "rejected",
                        "detail": f"bucket {bucket!r} error {rate:.4f} exceeds "
                        f"{thresholds['bucket_syllable_error_max']}",
                    }
                )

    contrast_judgments = int(contrast.get("judgments", 0))
    if contrast_judgments == 0:
        inconclusive.append("contrast_no_data")
        reasons.append(
            {
                "code": "contrast_no_data",
                "kind": "inconclusive",
                "detail": "no contrast judgments were supplied",
            }
        )
    else:
        accuracy = float(contrast.get("accuracy", 0.0))
        if accuracy < float(thresholds["contrast_accuracy_min"]):
            rejects.append("contrast_accuracy")
            reasons.append(
                {
                    "code": "contrast_accuracy",
                    "kind": "rejected",
                    "detail": f"contrast accuracy {accuracy:.4f} is below "
                    f"{thresholds['contrast_accuracy_min']} "
                    f"({contrast.get('correct')}/{contrast_judgments})",
                }
            )
        if contrast.get("systematic_critical_errors"):
            rejects.append("systematic_critical_error")
            reasons.append(
                {
                    "code": "systematic_critical_error",
                    "kind": "rejected",
                    "detail": f"{len(contrast['systematic_critical_errors'])} confirmed "
                    "systematic critical-contrast errors",
                }
            )

    if int(naturalness.get("count", 0)) == 0 or naturalness.get("mean") is None:
        inconclusive.append("naturalness_no_data")
        reasons.append(
            {
                "code": "naturalness_no_data",
                "kind": "inconclusive",
                "detail": "no candidate naturalness ratings were supplied",
            }
        )
    else:
        mean = float(naturalness["mean"])
        lower = naturalness.get("lower")
        if mean < float(thresholds["naturalness_mean_min"]):
            rejects.append("naturalness_mean")
            reasons.append(
                {
                    "code": "naturalness_mean",
                    "kind": "rejected",
                    "detail": f"mean naturalness {mean:.3f} is below "
                    f"{thresholds['naturalness_mean_min']}",
                }
            )
        elif lower is None:
            inconclusive.append("naturalness_interval_missing")
            reasons.append(
                {
                    "code": "naturalness_interval_missing",
                    "kind": "inconclusive",
                    "detail": "mean passes but no crossed-bootstrap lower bound was supplied",
                }
            )
        elif float(lower) < float(thresholds["naturalness_lower_bound_min"]):
            inconclusive.append("naturalness_lower_bound")
            reasons.append(
                {
                    "code": "naturalness_lower_bound",
                    "kind": "inconclusive",
                    "detail": f"95% lower bound {float(lower):.3f} is below "
                    f"{thresholds['naturalness_lower_bound_min']}; evidence is "
                    "insufficient for release",
                }
            )
        natural_mean = natural.get("mean")
        if natural_mean is None:
            inconclusive.append("natural_anchor_missing")
            reasons.append(
                {
                    "code": "natural_anchor_missing",
                    "kind": "inconclusive",
                    "detail": "no matched natural-speech anchor mean was supplied",
                }
            )
        else:
            gap = float(natural_mean) - mean
            if gap > float(thresholds["naturalness_gap_max"]):
                rejects.append("naturalness_gap")
                reasons.append(
                    {
                        "code": "naturalness_gap",
                        "kind": "rejected",
                        "detail": f"gap to natural speech {gap:.3f} exceeds "
                        f"{thresholds['naturalness_gap_max']}",
                    }
                )

    if catastrophic_outputs > 0:
        rejects.append("catastrophic_outputs")
        reasons.append(
            {
                "code": "catastrophic_outputs",
                "kind": "rejected",
                "detail": f"{catastrophic_outputs} catastrophic output(s) in the suite",
            }
        )

    missing_clauses = int(paragraphs.get("missing_clauses", 0))
    repeated_clauses = int(paragraphs.get("repeated_clauses", 0))
    language_switch_items = int(paragraphs.get("language_switch_items", 0))
    paragraph_items = int(paragraphs.get("items", 0))
    if paragraph_items == 0:
        inconclusive.append("paragraph_no_data")
        reasons.append(
            {
                "code": "paragraph_no_data",
                "kind": "inconclusive",
                "detail": "no continuous paragraphs were evaluated",
            }
        )
    elif missing_clauses or repeated_clauses or language_switch_items:
        rejects.append("paragraph_clauses")
        reasons.append(
            {
                "code": "paragraph_clauses",
                "kind": "rejected",
                "detail": f"paragraphs show {missing_clauses} missing, "
                f"{repeated_clauses} repeated clause(s) and "
                f"{language_switch_items} unintended language-switch item(s)",
            }
        )

    if panel_acceptance is None:
        inconclusive.append("panel_acceptance_missing")
        reasons.append(
            {
                "code": "panel_acceptance_missing",
                "kind": "inconclusive",
                "detail": "Jee and the native evaluation panel have not recorded acceptance",
            }
        )
    elif panel_acceptance is False:
        rejects.append("panel_rejected")
        reasons.append(
            {
                "code": "panel_rejected",
                "kind": "rejected",
                "detail": "the native panel did not accept the intended educational use",
            }
        )

    if rejects:
        verdict = "rejected"
    elif inconclusive:
        verdict = "inconclusive"
    else:
        verdict = "accepted"
    if verdict not in VERDICTS:  # pragma: no cover - defensive
        raise AssertionError(verdict)
    return {
        "verdict": verdict,
        "reasons": reasons,
        "failed_rules": rejects,
        "insufficient_evidence": inconclusive,
        "metrics": {
            "pooled_syllable_error": syllable.get("totals", {}).get("error_rate"),
            "contrast_accuracy": contrast.get("accuracy"),
            "contrast_denominator": contrast.get("judgments"),
            "naturalness_mean": naturalness.get("mean"),
            "naturalness_lower_bound": naturalness.get("lower"),
            "naturalness_gap": (
                None
                if natural.get("mean") is None or naturalness.get("mean") is None
                else float(natural["mean"]) - float(naturalness["mean"])
            ),
            "catastrophic_outputs": catastrophic_outputs,
            "missing_clauses": missing_clauses,
            "repeated_clauses": repeated_clauses,
            "language_switch_items": language_switch_items,
            "panel_acceptance": panel_acceptance,
        },
        "thresholds": dict(thresholds),
    }


__all__ = ["VERDICTS", "evaluate_release"]
