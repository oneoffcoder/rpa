from __future__ import annotations

import pytest

from rpa_tts.errors import InputError
from rpa_tts.evaluation.gate import evaluate_release

THRESHOLDS = {
    "pooled_syllable_error_max": 0.05,
    "bucket_syllable_error_max": 0.10,
    "contrast_accuracy_min": 0.95,
    "naturalness_mean_min": 4.0,
    "naturalness_lower_bound_min": 3.8,
    "naturalness_gap_max": 0.5,
}

BUCKETS = [
    "ordinary_sentences",
    "targeted_contrasts",
    "questions_numbers_names",
    "continuous_passages",
]


def _syllable(rate=0.02, bucket_rates=None):
    bucket_rates = bucket_rates or {bucket: rate for bucket in BUCKETS}
    return {
        "totals": {"error_rate": rate, "reference_syllables": 1000},
        "by_bucket": {
            bucket: {"error_rate": value, "reference_syllables": 250}
            for bucket, value in bucket_rates.items()
        },
    }


def _call(**overrides):
    arguments = {
        "thresholds": THRESHOLDS,
        "syllable": _syllable(),
        "contrast": {
            "accuracy": 0.97,
            "judgments": 400,
            "correct": 388,
            "systematic_critical_errors": [],
        },
        "naturalness": {"count": 900, "mean": 4.3, "lower": 4.1},
        "natural": {"mean": 4.6},
        "paragraphs": {"items": 20, "missing_clauses": 0, "repeated_clauses": 0},
        "catastrophic_outputs": 0,
        "panel_acceptance": True,
        "required_buckets": BUCKETS,
    }
    arguments.update(overrides)
    return evaluate_release(**arguments)


def test_all_rules_pass_yields_acceptance():
    verdict = _call()
    assert verdict["verdict"] == "accepted"
    assert verdict["failed_rules"] == []
    assert verdict["insufficient_evidence"] == []


def test_pooled_syllable_error_failure_rejects():
    verdict = _call(syllable=_syllable(rate=0.051))
    assert verdict["verdict"] == "rejected"
    assert "pooled_syllable_error" in verdict["failed_rules"]


def test_per_bucket_syllable_error_failure_rejects():
    verdict = _call(
        syllable=_syllable(
            bucket_rates={bucket: 0.02 for bucket in BUCKETS} | {"targeted_contrasts": 0.11}
        )
    )
    assert verdict["verdict"] == "rejected"
    assert any(rule.startswith("bucket_syllable_error") for rule in verdict["failed_rules"])


def test_missing_bucket_data_is_inconclusive():
    rates = {bucket: 0.02 for bucket in BUCKETS[:-1]}
    verdict = _call(syllable=_syllable(bucket_rates=rates))
    assert verdict["verdict"] == "inconclusive"
    assert any(code.startswith("bucket_no_data") for code in verdict["insufficient_evidence"])


def test_contrast_accuracy_failure_rejects():
    verdict = _call(
        contrast={
            "accuracy": 0.94,
            "judgments": 400,
            "correct": 376,
            "systematic_critical_errors": [],
        }
    )
    assert verdict["verdict"] == "rejected"
    assert "contrast_accuracy" in verdict["failed_rules"]


def test_systematic_critical_error_rejects_even_when_accuracy_passes():
    verdict = _call(
        contrast={
            "accuracy": 0.99,
            "judgments": 400,
            "correct": 396,
            "systematic_critical_errors": [{"expected": "hlub", "chosen": "hlu"}],
        }
    )
    assert verdict["verdict"] == "rejected"
    assert "systematic_critical_error" in verdict["failed_rules"]


def test_naturalness_mean_failure_rejects():
    verdict = _call(naturalness={"count": 900, "mean": 3.9, "lower": 3.7})
    assert verdict["verdict"] == "rejected"
    assert "naturalness_mean" in verdict["failed_rules"]


def test_naturalness_lower_bound_shortfall_is_inconclusive_not_rejected():
    verdict = _call(natural={"mean": 4.5}, naturalness={"count": 900, "mean": 4.05, "lower": 3.75})
    assert verdict["verdict"] == "inconclusive"
    assert "naturalness_lower_bound" in verdict["insufficient_evidence"]


def test_naturalness_gap_failure_rejects():
    verdict = _call(natural={"mean": 4.95}, naturalness={"count": 900, "mean": 4.4, "lower": 4.2})
    assert verdict["verdict"] == "rejected"
    assert "naturalness_gap" in verdict["failed_rules"]


def test_catastrophic_outputs_reject():
    verdict = _call(catastrophic_outputs=2)
    assert verdict["verdict"] == "rejected"
    assert "catastrophic_outputs" in verdict["failed_rules"]


def test_paragraph_clause_failures_reject():
    verdict = _call(paragraphs={"items": 20, "missing_clauses": 1, "repeated_clauses": 0})
    assert verdict["verdict"] == "rejected"
    assert "paragraph_clauses" in verdict["failed_rules"]


def test_paragraph_language_switch_rejects():
    verdict = _call(
        paragraphs={
            "items": 20,
            "missing_clauses": 0,
            "repeated_clauses": 0,
            "language_switch_items": 1,
        }
    )
    assert verdict["verdict"] == "rejected"
    assert "paragraph_clauses" in verdict["failed_rules"]
    assert verdict["metrics"]["language_switch_items"] == 1


def test_missing_panel_acceptance_is_inconclusive():
    verdict = _call(panel_acceptance=None)
    assert verdict["verdict"] == "inconclusive"
    assert "panel_acceptance_missing" in verdict["insufficient_evidence"]


def test_panel_rejection_rejects():
    verdict = _call(panel_acceptance=False)
    assert verdict["verdict"] == "rejected"
    assert "panel_rejected" in verdict["failed_rules"]


def test_missing_natural_anchor_is_inconclusive():
    verdict = _call(natural={})
    assert verdict["verdict"] == "inconclusive"
    assert "natural_anchor_missing" in verdict["insufficient_evidence"]


def test_no_transcription_data_is_inconclusive():
    verdict = _call(
        syllable={"totals": {"error_rate": 0.0, "reference_syllables": 0}, "by_bucket": {}}
    )
    assert verdict["verdict"] == "inconclusive"
    assert "no_transcription_data" in [reason["code"] for reason in verdict["reasons"]]


def test_missing_thresholds_are_an_input_error():
    with pytest.raises(InputError, match="thresholds"):
        _call(thresholds={"pooled_syllable_error_max": 0.05})
