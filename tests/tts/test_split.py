from __future__ import annotations

import pytest

from rpa_tts.data.split import assign_splits
from rpa_tts.errors import InputError


def _rows(rows_factory, count, **overrides):
    rows = rows_factory(count, **overrides)
    for index, row in enumerate(rows, 1):
        row["group_id"] = f"passage-{(index - 1) // 2:03d}"
        row["id"] = f"u{index:04d}"
    return rows


def test_groups_are_never_split(rows_factory):
    rows = _rows(rows_factory, 12)
    assigned, summary = assign_splits(rows, seed=3407, dev_fraction=0.1, test_fraction=0.2)
    seen: dict[str, str] = {}
    for row in assigned:
        seen.setdefault(row["group_id"], row["split"])
        assert seen[row["group_id"]] == row["split"]
    assert summary["groups"] == 6
    assert summary["row_counts"]["train"] > 0


def test_assignment_is_deterministic_and_seeded(rows_factory):
    rows = _rows(rows_factory, 12)
    first, summary_one = assign_splits(rows, seed=3407)
    second, summary_two = assign_splits(list(reversed(rows)), seed=3407)
    assert summary_one["assignment_sha256"] == summary_two["assignment_sha256"]
    assert [(row["id"], row["split"]) for row in first] == [
        (row["id"], row["split"]) for row in second
    ]


def test_bucket_quotas_leave_a_training_group(rows_factory):
    rows = _rows(rows_factory, 12)
    _, summary = assign_splits(rows, seed=11, dev_fraction=0.2, test_fraction=0.25)
    bucket = summary["buckets"]["ordinary_sentences"]
    assert bucket["test"] == 1
    assert bucket["dev"] == 1
    assert bucket["train"] == 4


def test_small_bucket_warns_instead_of_inventing_coverage(rows_factory):
    rows = _rows(rows_factory, 2)
    _, summary = assign_splits(rows, seed=3407, dev_fraction=0.1, test_fraction=0.2)
    assert any("no dev groups" in warning for warning in summary["warnings"])


def test_missing_group_id_is_rejected(rows_factory):
    rows = _rows(rows_factory, 4)
    rows[0]["group_id"] = ""
    with pytest.raises(InputError, match="no group_id"):
        assign_splits(rows)


def test_group_spanning_buckets_is_rejected(rows_factory):
    rows = _rows(rows_factory, 4)
    rows[1]["group_id"] = rows[0]["group_id"]
    rows[1]["bucket"] = "questions_numbers_names"
    with pytest.raises(InputError, match="cannot span buckets"):
        assign_splits(rows)


def test_duplicate_row_id_is_rejected(rows_factory):
    rows = _rows(rows_factory, 4)
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(InputError, match="duplicate row id"):
        assign_splits(rows)


def test_impossible_fractions_are_rejected(rows_factory):
    rows = _rows(rows_factory, 10)
    with pytest.raises(InputError, match="leave room"):
        assign_splits(rows, dev_fraction=0.5, test_fraction=0.5)
    with pytest.raises(InputError, match="between|must be in"):
        assign_splits(rows, dev_fraction=-0.1)


def test_empty_corpus_is_rejected():
    with pytest.raises(InputError, match="empty corpus"):
        assign_splits([])
