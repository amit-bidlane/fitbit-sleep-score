"""Tests for feature engineering and model-adjacent data transforms."""

from __future__ import annotations

import asyncio
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineer import SleepFeatureEngineer, engineer_sleep_features
from src.models.anomaly_detector import SleepAnomalyDetector, detect_sleep_anomalies
from src.models.score_calculator import SleepScoreCalculator, calculate_sleep_score
from src.models.sleep_classifier import (
    SleepClassifierInference,
    SleepDataset,
    SleepStageBiLSTM,
    SleepStageTrainer,
)


def test_engineer_sleep_features_computes_requested_sleep_metrics():
    sleep_frame = pd.DataFrame(
        [
            {
                "logId": 1001,
                "dateOfSleep": "2026-04-03",
                "raw_minutesAsleep": 420,
                "raw_minutesAwake": 60,
                "raw_timeInBed": 480,
                "raw_minutesToFallAsleep": 15,
                "raw_awakeningsCount": 2,
                "raw_levels_summary_deep_minutes": 90,
                "raw_levels_summary_rem_minutes": 100,
                "raw_levels_summary_light_minutes": 230,
                "raw_levels_summary_wake_minutes": 60,
            }
        ]
    )

    hrv_frame = pd.DataFrame([{"dateTime": "2026-04-03", "value_dailyRmssd": 52.5}])
    heart_frame = pd.DataFrame([{"dateTime": "2026-04-03", "restingHeartRate": 58}])
    spo2_frame = pd.DataFrame([{"dateTime": "2026-04-03", "value_avg": 97.4}])

    features = engineer_sleep_features(
        sleep_data=sleep_frame,
        hrv_data=hrv_frame,
        heart_rate_data=heart_frame,
        spo2_data=spo2_frame,
    )

    row = features.iloc[0]
    assert row["total_sleep_minutes"] == 420
    assert round(row["deep_sleep_pct"], 2) == 21.43
    assert round(row["rem_sleep_pct"], 2) == 23.81
    assert round(row["light_sleep_pct"], 2) == 54.76
    assert round(row["awake_pct"], 2) == 12.50
    assert round(row["sleep_efficiency"], 2) == 87.50
    assert row["sleep_onset_latency"] == 15
    assert row["number_of_awakenings"] == 2
    assert row["avg_hrv"] == 52.5
    assert row["resting_hr"] == 58
    assert row["avg_spo2"] == 97.4
    assert row["day_of_week"] == 4
    assert row["is_weekend"] == 0
    assert row["sleep_continuity_score"] == 73.0


def test_engineer_sleep_features_supports_weekend_and_fallback_columns():
    sleep_payload = {
        "sleep": [
            {
                "logId": 1002,
                "dateOfSleep": "2026-04-04",
                "minutesAsleep": 450,
                "minutesAwake": 30,
                "timeInBed": 500,
                "minutesToFallAsleep": 10,
                "awakeningsCount": 1,
                "levels": {
                    "summary": {
                        "deep": {"minutes": 120},
                        "rem": {"minutes": 90},
                        "light": {"minutes": 240},
                        "wake": {"minutes": 30, "count": 1},
                    }
                },
            }
        ]
    }

    features = SleepFeatureEngineer().engineer(sleep_payload)

    row = features.iloc[0]
    assert row["total_sleep_minutes"] == 450
    assert round(row["deep_sleep_pct"], 2) == 26.67
    assert round(row["rem_sleep_pct"], 2) == 20.00
    assert round(row["light_sleep_pct"], 2) == 53.33
    assert round(row["awake_pct"], 2) == 6.00
    assert round(row["sleep_efficiency"], 2) == 90.00
    assert row["number_of_awakenings"] == 1
    assert row["day_of_week"] == 5
    assert row["is_weekend"] == 1


def test_engineer_sleep_features_aggregates_duplicate_daily_metrics():
    sleep_frame = pd.DataFrame(
        [
            {
                "logId": 1003,
                "dateOfSleep": "2026-04-05",
                "raw_minutesAsleep": 400,
                "raw_minutesAwake": 50,
                "raw_timeInBed": 460,
                "raw_levels_summary_deep_minutes": 80,
                "raw_levels_summary_rem_minutes": 100,
                "raw_levels_summary_light_minutes": 220,
            }
        ]
    )
    hrv_frame = pd.DataFrame(
        [
            {"dateTime": "2026-04-05", "value_dailyRmssd": 40.0},
            {"dateTime": "2026-04-05", "value_dailyRmssd": 44.0},
        ]
    )
    heart_frame = pd.DataFrame(
        [
            {"dateTime": "2026-04-05", "restingHeartRate": 60},
            {"dateTime": "2026-04-05", "restingHeartRate": 64},
        ]
    )
    spo2_frame = pd.DataFrame(
        [
            {"dateTime": "2026-04-05", "value_avg": 96.0},
            {"dateTime": "2026-04-05", "value_avg": 98.0},
        ]
    )

    features = engineer_sleep_features(
        sleep_data=sleep_frame,
        hrv_data=hrv_frame,
        heart_rate_data=heart_frame,
        spo2_data=spo2_frame,
    )

    row = features.iloc[0]
    assert row["avg_hrv"] == 42.0
    assert row["resting_hr"] == 62.0
    assert row["avg_spo2"] == 97.0


def test_score_calculator_returns_component_scores_and_quality_label():
    features = pd.DataFrame(
        [
            {
                "total_sleep_minutes": 465,
                "deep_sleep_pct": 19.0,
                "rem_sleep_pct": 22.0,
                "awake_pct": 6.0,
                "sleep_efficiency": 92.0,
                "sleep_onset_latency": 8.0,
                "number_of_awakenings": 1.0,
                "sleep_continuity_score": 88.0,
                "avg_hrv": 56.0,
                "resting_hr": 56.0,
                "avg_spo2": 98.0,
            }
        ]
    )

    scored = calculate_sleep_score(features)

    row = scored.iloc[0]
    assert row["duration_score"] == 100.0
    assert row["deep_score"] == 87.5
    assert row["rem_score"] == 100.0
    assert row["continuity_score"] == 88.0
    assert row["recovery_score"] == 84.9
    assert round(row["final_score"], 2) == 92.7
    assert row["label"] == "Excellent"


def test_score_calculator_gracefully_scores_lower_quality_sleep():
    features = {
        "total_sleep_minutes": 300,
        "deep_sleep_pct": 8.0,
        "rem_sleep_pct": 10.0,
        "awake_pct": 20.0,
        "sleep_efficiency": 70.0,
        "sleep_onset_latency": 35.0,
        "number_of_awakenings": 5.0,
        "avg_hrv": 24.0,
        "resting_hr": 78.0,
        "avg_spo2": 93.0,
    }

    row = SleepScoreCalculator().calculate(features).iloc[0]

    assert round(row["continuity_score"], 2) == 61.5
    assert round(row["recovery_score"], 2) == 23.5
    assert round(row["final_score"], 2) == 29.0
    assert row["label"] == "Poor"
    assert len(row["recommendations"]) == 5


def test_score_calculator_generates_rule_based_recommendations():
    features = pd.DataFrame(
        [
            {
                "total_sleep_minutes": 300,
                "deep_sleep_pct": 12.0,
                "rem_sleep_pct": 18.0,
                "number_of_awakenings": 6.0,
                "avg_spo2": 92.0,
                "sleep_onset_latency": 45.0,
                "sleep_efficiency": 70.0,
            }
        ]
    )

    row = calculate_sleep_score(features).iloc[0]
    recommendation_titles = {item["title"] for item in row["recommendations"]}

    assert recommendation_titles == {
        "Reduce pre-bed alcohol",
        "Keep a consistent sleep schedule",
        "Review bedroom temperature",
        "Discuss low overnight oxygen levels",
        "Add a wind-down routine",
        "Increase total sleep time",
    }


def test_anomaly_detector_returns_expected_columns_and_flags_outlier():
    baseline = pd.DataFrame(
        [
            {
                "total_sleep_minutes": 430,
                "deep_sleep_pct": 20.0,
                "rem_sleep_pct": 22.0,
                "awake_pct": 8.0,
                "sleep_efficiency": 90.0,
                "sleep_onset_latency": 12.0,
                "number_of_awakenings": 1.0,
                "avg_hrv": 50.0,
                "resting_hr": 58.0,
                "avg_spo2": 97.0,
                "sleep_continuity_score": 84.0,
            },
            {
                "total_sleep_minutes": 440,
                "deep_sleep_pct": 21.0,
                "rem_sleep_pct": 23.0,
                "awake_pct": 7.0,
                "sleep_efficiency": 91.0,
                "sleep_onset_latency": 10.0,
                "number_of_awakenings": 1.0,
                "avg_hrv": 52.0,
                "resting_hr": 57.0,
                "avg_spo2": 97.0,
                "sleep_continuity_score": 86.0,
            },
            {
                "total_sleep_minutes": 420,
                "deep_sleep_pct": 19.0,
                "rem_sleep_pct": 21.0,
                "awake_pct": 9.0,
                "sleep_efficiency": 89.0,
                "sleep_onset_latency": 14.0,
                "number_of_awakenings": 2.0,
                "avg_hrv": 49.0,
                "resting_hr": 59.0,
                "avg_spo2": 96.0,
                "sleep_continuity_score": 80.0,
            },
        ]
    )
    candidate_rows = pd.concat(
        [
            baseline.iloc[[0]],
            pd.DataFrame(
                [
                    {
                        "total_sleep_minutes": 220,
                        "deep_sleep_pct": 5.0,
                        "rem_sleep_pct": 8.0,
                        "awake_pct": 28.0,
                        "sleep_efficiency": 58.0,
                        "sleep_onset_latency": 55.0,
                        "number_of_awakenings": 7.0,
                        "avg_hrv": 18.0,
                        "resting_hr": 84.0,
                        "avg_spo2": 91.0,
                        "sleep_continuity_score": 32.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    detected = detect_sleep_anomalies(candidate_rows, baseline_data=baseline, feature_columns=baseline.columns.tolist())

    assert {"anomaly_score", "is_anomaly", "anomaly_reason"}.issubset(detected.columns)
    assert bool(detected.iloc[0]["is_anomaly"]) is False
    assert detected.iloc[0]["anomaly_reason"] == "normal_sleep_pattern"
    assert bool(detected.iloc[1]["is_anomaly"]) is True
    assert detected.iloc[1]["anomaly_score"] > detected.iloc[0]["anomaly_score"]
    assert "low sleep_efficiency" in detected.iloc[1]["anomaly_reason"]


def test_anomaly_detector_handles_insufficient_baseline_data():
    features = [{"total_sleep_minutes": 430, "sleep_efficiency": 90.0}]

    detected = SleepAnomalyDetector().detect(features)

    row = detected.iloc[0]
    assert row["anomaly_score"] == 0.0
    assert bool(row["is_anomaly"]) is False
    assert row["anomaly_reason"] == "insufficient_baseline_data"


def test_sleep_dataset_validates_shape_and_returns_items():
    sequences = np.random.rand(4, 10, 6).astype("float32")
    labels = np.array([0, 1, 2, 3], dtype="int64")

    dataset = SleepDataset(sequences, labels)

    assert len(dataset) == 4
    sequence, label = dataset[1]
    assert tuple(sequence.shape) == (10, 6)
    assert int(label.item()) == 1

    with pytest.raises(ValueError, match="sequence_length must be 10"):
        SleepDataset(np.random.rand(4, 8, 6).astype("float32"), labels)


def test_sleep_stage_bilstm_outputs_four_class_logits():
    model = SleepStageBiLSTM()
    inputs = SleepDataset(np.random.rand(3, 10, 6).astype("float32")).sequences

    outputs = model(inputs)

    assert tuple(outputs.shape) == (3, 4)
    assert model.output_size == 256


def test_sleep_classifier_inference_loads_checkpoint_and_predicts():
    checkpoint_dir = Path("tests_artifacts") / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer = SleepStageTrainer(
        epochs=1,
        batch_size=2,
        patience=1,
        checkpoint_dir=str(checkpoint_dir),
    )
    checkpoint_path = trainer.save_checkpoint()

    inference = SleepClassifierInference(checkpoint_path=checkpoint_path)
    model = inference.load_model()

    sequence = np.random.rand(10, 6).astype("float32")
    probabilities = inference.predict_proba(sequence)
    prediction = inference.predict(sequence)
    batch_predictions = inference.batch_predict(np.random.rand(2, 10, 6).astype("float32"))

    assert model.output_size == 256
    assert Path(checkpoint_path).exists()
    assert len(probabilities) == 4
    assert round(sum(probabilities), 5) == 1.0
    assert prediction in {0, 1, 2, 3}
    assert len(batch_predictions) == 2


def test_score_calculator_store_recommendations_uses_crud(monkeypatch: pytest.MonkeyPatch):
    calculator = SleepScoreCalculator()
    created_payloads: list[dict[str, object]] = []

    async def fake_create_recommendation(session, **recommendation_data):
        created_payloads.append(recommendation_data)
        return recommendation_data

    import src.database.crud as crud_module

    monkeypatch.setattr(crud_module, "create_recommendation", fake_create_recommendation)

    records = asyncio.run(
        calculator.store_recommendations(
            session=object(),
            user_id=7,
            sleep_score_id=12,
            recommendation_date=date(2026, 4, 5),
            sleep_features={
                "total_sleep_minutes": 300,
                "deep_sleep_pct": 12.0,
                "rem_sleep_pct": 18.0,
                "number_of_awakenings": 6.0,
                "avg_spo2": 92.0,
                "sleep_onset_latency": 45.0,
            },
        )
    )

    assert len(records) == 6
    assert len(created_payloads) == 6
    assert all(payload["user_id"] == 7 for payload in created_payloads)
    assert all(payload["sleep_score_id"] == 12 for payload in created_payloads)
    assert all(payload["recommendation_date"] == date(2026, 4, 5) for payload in created_payloads)
