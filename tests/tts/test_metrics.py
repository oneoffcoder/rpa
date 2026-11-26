from __future__ import annotations

import pytest

from rpa_tts.errors import InputError, UnsupportedInputError
from rpa_tts.evaluation.metrics import (
    contrast_summary,
    listener_syllables,
    naturalness_summary,
    reference_syllables,
    score_transcriptions,
    syllable_ops,
)


@pytest.mark.parametrize(
    "reference, hypothesis, expected",
    [
        (["kuv"], ["kuv"], (0, 0, 0, 0)),
        (["kuv"], ["koj"], (1, 1, 0, 0)),
        (["kuv", "hlub"], ["kuv"], (1, 0, 1, 0)),
        (["kuv"], ["kuv", "hlub"], (1, 0, 0, 1)),
        (["a", "b", "c"], ["a", "x", "c", "d"], (2, 1, 0, 1)),
        ([], [], (0, 0, 0, 0)),
        ([], ["a", "b"], (2, 0, 0, 2)),
    ],
)
def test_syllable_ops_counts(reference, hypothesis, expected):
    assert syllable_ops(reference, hypothesis) == expected


def test_syllable_ops_is_deterministic_for_ambiguous_edit_paths():
    reference = ["a", "b"]
    hypothesis = ["c", "d"]
    assert syllable_ops(reference, hypothesis) == syllable_ops(reference, hypothesis)


def test_listener_tokenizer_keeps_unknown_spellings_for_scoring():
    assert listener_syllables("Kuv blorp, KOJ!") == ["kuv", "blorp", "koj"]


def test_reference_tokenizer_requires_valid_rpa():
    assert reference_syllables("kuv hlub koj") == ["kuv", "hlub", "koj"]
    with pytest.raises(UnsupportedInputError):
        reference_syllables("kuv blorp")


def test_pooled_error_rate_and_empty_output_denominator():
    records = [
        {
            "utterance_id": "u1",
            "listener_id": "l1",
            "bucket": "ordinary_sentences",
            "reference": "kuv hlub koj",
            "hypothesis": "kuv hlub",
        },
        {
            "utterance_id": "u2",
            "listener_id": "l2",
            "bucket": "questions_numbers_names",
            "reference": "nws yog ib",
            "hypothesis": "",
        },
    ]
    report = score_transcriptions(records)
    assert report["totals"]["reference_syllables"] == 6
    assert report["totals"]["deletions"] == 4
    assert report["totals"]["errors"] == 4
    assert report["totals"]["empty_outputs"] == 1
    assert report["pooled_error_rate"] == pytest.approx(4 / 6, abs=1e-6)
    assert report["by_bucket"]["ordinary_sentences"]["errors"] == 1
    assert report["by_bucket"]["questions_numbers_names"]["errors"] == 3
    assert report["totals"]["utterances"] == 2


def test_unknown_hypothesis_word_counts_as_a_substitution():
    report = score_transcriptions(
        [
            {
                "utterance_id": "u1",
                "listener_id": "l1",
                "bucket": "ordinary_sentences",
                "reference": "kuv",
                "hypothesis": "blorp",
            }
        ]
    )
    assert report["totals"]["substitutions"] == 1
    assert report["totals"]["errors"] == 1


def test_multi_listener_records_are_pooled_not_averaged():
    records = [
        {
            "utterance_id": "u1",
            "listener_id": "l1",
            "bucket": "ordinary_sentences",
            "reference": "kuv hlub koj",
            "hypothesis": "kuv hlub koj",
        },
        {
            "utterance_id": "u1",
            "listener_id": "l2",
            "bucket": "ordinary_sentences",
            "reference": "kuv hlub koj",
            "hypothesis": "",
        },
    ]
    report = score_transcriptions(records)
    assert report["pooled_error_rate"] == pytest.approx(3 / 6)
    assert report["by_listener"]["l2"]["error_rate"] == 1.0


def test_missing_reference_is_an_input_error():
    with pytest.raises(InputError, match="missing"):
        score_transcriptions(
            [{"utterance_id": "u1", "listener_id": "l1", "bucket": "b", "hypothesis": "kuv"}]
        )
    with pytest.raises(InputError, match="no transcription records"):
        score_transcriptions([])


def test_naturalness_summary_uses_integer_scale():
    records = [
        {"utterance_id": "u1", "listener_id": "l1", "condition": "candidate", "score": 4},
        {"utterance_id": "u2", "listener_id": "l2", "condition": "candidate", "score": 5},
        {"utterance_id": "u1", "listener_id": "l1", "condition": "natural", "score": 5},
    ]
    summary = naturalness_summary(records, condition="candidate")
    assert summary["count"] == 2
    assert summary["mean"] == 4.5
    assert summary["distribution"] == {"4": 1, "5": 1}
    with pytest.raises(InputError, match="integer score"):
        naturalness_summary(
            [{"utterance_id": "u1", "listener_id": "l1", "condition": "candidate", "score": 4.5}]
        )
    with pytest.raises(InputError, match="1 to 5"):
        naturalness_summary(
            [{"utterance_id": "u1", "listener_id": "l1", "condition": "candidate", "score": 6}]
        )


def test_contrast_accuracy_keeps_ties_and_denominator():
    records = [
        {"utterance_id": "u1", "listener_id": "l1", "expected": "hlub", "chosen": "hlub"},
        {"utterance_id": "u2", "listener_id": "l1", "expected": "hlub", "chosen": "hlub"},
        {"utterance_id": "u3", "listener_id": "l2", "expected": "hlub", "chosen": None},
    ]
    report = contrast_summary(records)
    assert report["judgments"] == 3
    assert report["correct"] == 2
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["ties"] == 1
    assert report["confusion"]["hlub->no_response"] == 1
    assert report["blocks_release"] is False


def test_systematic_critical_error_blocks_release():
    records = [
        {
            "utterance_id": "u1",
            "listener_id": "l1",
            "expected": "hlub",
            "chosen": "hlub",
            "critical": True,
            "adjudicated": True,
            "confirmed_by": ["a1", "a2"],
        },
        {
            "utterance_id": "u2",
            "listener_id": "l1",
            "expected": "hlub",
            "chosen": "hlu",
            "critical": True,
            "adjudicated": True,
            "confirmed_by": ["a1", "a2"],
        },
        {
            "utterance_id": "u3",
            "listener_id": "l2",
            "expected": "hlub",
            "chosen": "hlu",
            "critical": True,
            "adjudicated": True,
            "confirmed_by": ["a1", "a2"],
        },
    ]
    report = contrast_summary(records)
    assert len(report["systematic_critical_errors"]) == 1
    entry = report["systematic_critical_errors"][0]
    assert entry["distinct_utterances"] == 2
    assert entry["adjudicators"] == ["a1", "a2"]
    assert report["blocks_release"] is True


def test_single_confirmed_error_is_counted_but_not_systematic():
    records = [
        {
            "utterance_id": "u2",
            "listener_id": "l1",
            "expected": "hlub",
            "chosen": "hlu",
            "critical": True,
            "adjudicated": True,
            "confirmed_by": ["a1", "a2"],
        }
    ]
    report = contrast_summary(records)
    assert report["systematic_critical_errors"] == []
    assert report["blocks_release"] is False
    assert report["accuracy"] == 0.0
