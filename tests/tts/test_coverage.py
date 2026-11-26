from __future__ import annotations

from rpa_tts.data.contract import ContrastCell
from rpa_tts.data.coverage import coverage_report, render_coverage_markdown

HLU = ContrastCell(id="C-01", type="onset_tone", consonant="hl", vowel="u", tone="b", attested=True)
AIS = ContrastCell(
    id="C-02", type="vowel_tone", consonant=None, vowel="ai", tone="s", attested=True
)
UNATTESTED = ContrastCell(
    id="C-03", type="onset_tone", consonant="plh", vowel="aw", tone="g", attested=False
)


def _rows(rows_factory, count, *, split="train", text="kuv hlub koj", start=1):
    rows = rows_factory(count)
    for offset, row in enumerate(rows):
        row["id"] = f"u{start + offset:04d}"
        row["split"] = split
        row["transcript_raw"] = text
        from rpa_tts.text.normalize import normalize_rpa

        row["transcript_normalized"] = normalize_rpa(text).normalized
    return rows


def test_cell_counting_uses_distinct_utterances(rows_factory):
    rows = _rows(rows_factory, 5)
    rows[0]["transcript_raw"] = "hlub hlub hlub"
    from rpa_tts.text.normalize import normalize_rpa

    rows[0]["transcript_normalized"] = normalize_rpa(rows[0]["transcript_raw"]).normalized
    report = coverage_report(rows, [HLU], min_examples=5)
    cell = report["cells"][0]
    assert cell["splits"]["train"]["examples"] == 5
    assert cell["splits"]["train"]["occurrences"] == 7
    assert cell["status"] == "ok"
    assert cell["distinct_forms"] == ["hlub"]


def test_missing_and_underfilled_cells_are_reported(rows_factory):
    underfilled_rows = _rows(rows_factory, 3)
    report = coverage_report(underfilled_rows, [HLU, AIS], min_examples=5)
    by_id = {row["id"]: row for row in report["cells"]}
    assert by_id["C-01"]["status"] == "underfilled"
    assert by_id["C-02"]["status"] == "missing"
    assert report["missing"] == ["C-02"]
    assert report["underfilled"] == ["C-01"]
    assert report["screening_passed"] is False


def test_held_out_context_is_required_for_the_screen(rows_factory):
    rows = _rows(rows_factory, 5)
    report = coverage_report(rows, [HLU], min_examples=5)
    assert report["cells"][0]["splits"]["test"]["examples"] == 0
    assert report["cells"][0]["screening_target_met"] is False
    assert report["screening_passed"] is False


def test_test_split_examples_satisfy_the_screen(rows_factory):
    rows = _rows(rows_factory, 5) + _rows(rows_factory, 1, split="test", start=100)
    report = coverage_report(rows, [HLU], min_examples=5)
    assert report["cells"][0]["splits"]["test"]["examples"] == 1
    assert report["screening_passed"] is True


def test_unattested_cells_are_excluded_not_counted(rows_factory):
    rows = _rows(rows_factory, 5)
    report = coverage_report(rows, [UNATTESTED], min_examples=5)
    assert report["cells"][0]["status"] == "unattested_excluded"
    assert report["cells"][0]["screening_target_met"] is False
    # No attested cell means no screen has passed; an empty contract is incomplete.
    assert report["screening_passed"] is False


def test_observed_component_counts_are_recorded(rows_factory):
    rows = _rows(rows_factory, 2)
    report = coverage_report(rows, [HLU])
    assert report["observed_onset_tone"]["hl_b"] == 2
    assert report["observed_vowel_tone"]["u_b"] == 2


def test_markdown_ledger_lists_cells_and_status(rows_factory):
    rows = _rows(rows_factory, 5)
    report = coverage_report(rows, [HLU, AIS], min_examples=5)
    markdown = render_coverage_markdown(report)
    assert "| C-01 |" in markdown
    assert "underfilled" in markdown or "missing" in markdown
    assert "Coverage ledger" in markdown


def test_min_examples_must_be_positive(rows_factory):
    import pytest

    with pytest.raises(ValueError, match="at least 1"):
        coverage_report(_rows(rows_factory, 1), [HLU], min_examples=0)
