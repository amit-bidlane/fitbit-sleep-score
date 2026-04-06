"""Tests for the Fitbit sleep preprocessor."""

from __future__ import annotations

import pandas as pd

from src.data.preprocessor import SleepPreprocessor, preprocess_sleep_data


def test_preprocess_sleep_data_cleans_missing_values_and_scales():
    raw_payload = {
        "sleep": [
            {
                "logId": 101,
                "dateOfSleep": "2026-04-01",
                "startTime": "2026-03-31T23:00:00.000",
                "endTime": "2026-04-01T07:00:00.000",
                "minutesAsleep": 420,
                "minutesAwake": 45,
                "timeInBed": 480,
                "efficiency": 87,
            },
            {
                "logId": 102,
                "dateOfSleep": "2026-04-02",
                "startTime": "2026-04-01T23:15:00.000",
                "endTime": "2026-04-02T07:05:00.000",
                "minutesAsleep": None,
                "minutesAwake": 50,
                "timeInBed": 470,
                "efficiency": None,
            },
            {
                "logId": 103,
                "dateOfSleep": "2026-04-03",
                "startTime": "2026-04-02T23:30:00.000",
                "endTime": "2026-04-03T07:25:00.000",
                "minutesAsleep": 430,
                "minutesAwake": None,
                "timeInBed": 475,
                "efficiency": 90,
            },
        ]
    }

    processed = preprocess_sleep_data(raw_payload)

    assert len(processed) == 3
    assert processed.isna().sum().sum() == 0
    assert pd.api.types.is_datetime64_any_dtype(processed["startTime"])
    assert processed["logId"].tolist() == [101, 102, 103]

    scaled_columns = ["minutesAsleep", "minutesAwake", "timeInBed", "efficiency", "durationMinutes"]
    for column in scaled_columns:
        assert processed[column].between(0, 1).all()


def test_preprocess_sleep_data_removes_iqr_outlier_rows():
    raw_payload = {
        "sleep": [
            {
                "logId": 201,
                "dateOfSleep": "2026-04-01",
                "startTime": "2026-03-31T23:00:00.000",
                "endTime": "2026-04-01T07:00:00.000",
                "minutesAsleep": 420,
                "minutesAwake": 35,
                "timeInBed": 455,
                "efficiency": 92,
            },
            {
                "logId": 202,
                "dateOfSleep": "2026-04-02",
                "startTime": "2026-04-01T23:10:00.000",
                "endTime": "2026-04-02T07:05:00.000",
                "minutesAsleep": 425,
                "minutesAwake": 40,
                "timeInBed": 465,
                "efficiency": 91,
            },
            {
                "logId": 203,
                "dateOfSleep": "2026-04-03",
                "startTime": "2026-04-02T23:20:00.000",
                "endTime": "2026-04-03T07:10:00.000",
                "minutesAsleep": 430,
                "minutesAwake": 42,
                "timeInBed": 470,
                "efficiency": 90,
            },
            {
                "logId": 204,
                "dateOfSleep": "2026-04-04",
                "startTime": "2026-04-03T20:00:00.000",
                "endTime": "2026-04-04T12:00:00.000",
                "minutesAsleep": 1000,
                "minutesAwake": 300,
                "timeInBed": 1200,
                "efficiency": 20,
            },
        ]
    }

    processed = SleepPreprocessor().preprocess(raw_payload)

    assert processed["logId"].tolist() == [201, 202, 203]
    assert 204 not in processed["logId"].tolist()


def test_preprocess_sleep_data_accepts_dataframe_input():
    raw_frame = pd.DataFrame(
        [
            {
                "logId": 301,
                "dateOfSleep": "2026-04-01",
                "startTime": "2026-03-31T23:00:00.000",
                "endTime": "2026-04-01T07:00:00.000",
                "minutesAsleep": 420,
                "minutesAwake": 45,
                "timeInBed": 480,
                "efficiency": 87,
            },
            {
                "logId": 302,
                "dateOfSleep": "2026-04-02",
                "startTime": "2026-04-01T23:00:00.000",
                "endTime": "2026-04-02T07:15:00.000",
                "minutesAsleep": 435,
                "minutesAwake": 40,
                "timeInBed": 495,
                "efficiency": 89,
            },
        ]
    )

    processed = SleepPreprocessor().preprocess(raw_frame)

    assert len(processed) == 2
    assert processed["logId"].tolist() == [301, 302]
    assert "sleepEfficiencyComputed" in processed.columns
