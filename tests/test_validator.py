"""Tests for submission validator integration."""

from pathlib import Path

from src.evaluation import run_validator


def test_sample_submission_valid():
    sample = Path(__file__).resolve().parent.parent / "data" / "raw" / "sample_submission.csv"
    ok, errors = run_validator(sample)
    assert ok, errors
