"""Tests for the anomaly detector evaluation script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "evaluate_anomaly", Path(__file__).resolve().parents[2] / "scripts" / "evaluate_anomaly.py"
)
evaluate_anomaly = importlib.util.module_from_spec(_SPEC)
sys.modules["evaluate_anomaly"] = evaluate_anomaly  # needed for dataclasses
_SPEC.loader.exec_module(evaluate_anomaly)


def test_counts_metrics():
    c = evaluate_anomaly.Counts(tp=8, fp=2, fn=2)
    assert c.precision == 0.8
    assert c.recall == 0.8
    assert round(c.f1, 4) == 0.8


def test_counts_handle_zero_division():
    c = evaluate_anomaly.Counts()
    assert (c.precision, c.recall, c.f1) == (0.0, 0.0, 0.0)


def test_evaluate_site_scores_only_unseen_days(tmp_path):
    counts = evaluate_anomaly.evaluate_site(
        "TEST-EVAL", "factory", seed=3, days=30, train_days=20, artifact_dir=str(tmp_path)
    )
    # Only the 10 held-out days (240 hours) are scored, never the training days.
    assert counts.tp + counts.fp + counts.fn <= 10 * 24
    assert 0.0 <= counts.recall <= 1.0


def test_detector_catches_most_injected_anomalies(tmp_path):
    counts = evaluate_anomaly.evaluate_site(
        "TEST-EVAL", "factory", seed=1, days=90, train_days=60, artifact_dir=str(tmp_path)
    )
    # Guards against regressions: scoring raw readings should catch nearly every spike.
    assert counts.recall >= 0.8


def test_raw_readings_catch_more_than_cleaned(tmp_path):
    raw = evaluate_anomaly.evaluate_site(
        "TEST-EVAL", "warehouse", seed=2, artifact_dir=str(tmp_path)
    )
    cleaned = evaluate_anomaly.evaluate_site(
        "TEST-EVAL", "warehouse", seed=2, cleaned_readings=True, artifact_dir=str(tmp_path)
    )
    assert raw.recall >= cleaned.recall


def test_format_table_markdown():
    table = evaluate_anomaly.format_table(
        {"All sites": evaluate_anomaly.Counts(tp=3, fp=1, fn=1)}, markdown=True
    )
    assert table.splitlines()[0].startswith("| Site |")
    assert "75%" in table
