from __future__ import annotations

import pytest

from rpa_tts.errors import InputError
from rpa_tts.evaluation.uncertainty import (
    crossed_bootstrap,
    mean_statistic,
    mean_with_interval,
    ratio_statistic,
    ratio_with_interval,
)


def _records(values_by_listener, scripts=("u1",), bucket="ordinary_sentences", field="score"):
    records = []
    for listener, value in values_by_listener.items():
        for script in scripts:
            records.append(
                {
                    "listener_id": listener,
                    "utterance_id": script,
                    "bucket": bucket,
                    field: value,
                }
            )
    return records


def test_constant_scores_produce_a_point_interval():
    records = _records({"l1": 4, "l2": 4}, scripts=("u1", "u2"))
    result = mean_with_interval(records, resamples=200, seed=7)
    assert result["estimate"] == 4
    assert result["lower"] == 4
    assert result["upper"] == 4
    assert result["valid_resamples"] == 200


def test_listener_resampling_uses_the_full_listener_set():
    records = _records({"l1": 1, "l2": 5})
    result = crossed_bootstrap(records, mean_statistic, resamples=300, seed=3, stratum_field=None)
    assert result["estimate"] == 3
    assert result["lower"] <= 1.0001
    assert result["upper"] >= 4.9999
    assert result["listeners"] == ["l1", "l2"]


def test_bootstrap_is_reproducible_for_a_recorded_seed():
    records = _records({"l1": 3, "l2": 4, "l3": 5}, scripts=("u1", "u2", "u3"))
    first = mean_with_interval(records, resamples=200, seed=99)
    second = mean_with_interval(records, resamples=200, seed=99)
    third = mean_with_interval(records, resamples=200, seed=100)
    assert (first["lower"], first["upper"]) == (second["lower"], second["upper"])
    assert first["seed"] == 99
    assert third["lower"] <= third["estimate"] <= third["upper"]


def test_scripts_are_resampled_within_strata():
    records = _records({"l1": 4, "l2": 5}, scripts=("u1", "u2"), bucket="ordinary_sentences")
    records += _records({"l1": 2, "l2": 3}, scripts=("u3", "u4"), bucket="targeted_contrasts")
    result = mean_with_interval(records, resamples=300, seed=11)
    assert result["strata"] == {"ordinary_sentences": 2, "targeted_contrasts": 2}
    assert result["estimate"] == pytest.approx(3.5)


def test_ratio_statistic_weights_by_denominator():
    records = [
        {"listener_id": "l1", "utterance_id": "u1", "bucket": "b", "errors": 1, "reference": 10},
        {"listener_id": "l2", "utterance_id": "u1", "bucket": "b", "errors": 0, "reference": 10},
        {"listener_id": "l1", "utterance_id": "u2", "bucket": "b", "errors": 2, "reference": 20},
        {"listener_id": "l2", "utterance_id": "u2", "bucket": "b", "errors": 0, "reference": 20},
    ]
    statistic = ratio_statistic("errors", "reference")
    assert statistic(records) == pytest.approx(3 / 60)
    result = ratio_with_interval(
        records, numerator_field="errors", denominator_field="reference", resamples=200, seed=5
    )
    assert result["estimate"] == pytest.approx(0.05)
    assert result["lower"] <= result["estimate"] <= result["upper"]


def test_single_listener_is_rejected_because_dependence_cannot_be_estimated():
    with pytest.raises(InputError, match="two distinct listeners"):
        mean_with_interval(_records({"l1": 4}, scripts=("u1", "u2")))


def test_too_few_resamples_are_rejected():
    with pytest.raises(InputError, match="at least 100"):
        mean_with_interval(_records({"l1": 4, "l2": 5}), resamples=50)


def test_empty_records_are_rejected():
    with pytest.raises(InputError, match="no records"):
        crossed_bootstrap([], mean_statistic)


def test_interval_covers_the_estimate_for_varied_data():
    records = _records({"l1": 3, "l2": 4, "l3": 4, "l4": 5}, scripts=("u1", "u2"))
    result = mean_with_interval(records, resamples=400, seed=17)
    assert result["lower"] <= result["estimate"] <= result["upper"]
    assert result["confidence"] == 0.95
    assert "percentile" in result["method"]
