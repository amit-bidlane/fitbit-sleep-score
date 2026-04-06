"""Synthetic sample data generation utilities for the Fitbit Sleep Score project."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SyntheticSleepDataGenerator:
    """Create realistic-enough synthetic nightly sleep data."""

    seed: int = 42
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the reproducible random generator."""

        self._rng = np.random.default_rng(self.seed)

    def generate(self, *, days: int = 14, end_date: date | None = None) -> dict[str, pd.DataFrame]:
        """Generate synthetic sessions, stages, biometrics, and trend data."""

        if days < 1:
            raise ValueError("days must be at least 1")

        final_date = end_date or date.today()
        records: list[dict[str, Any]] = []
        stages: list[dict[str, Any]] = []
        heart_rate_rows: list[dict[str, Any]] = []
        hrv_rows: list[dict[str, Any]] = []
        spo2_rows: list[dict[str, Any]] = []

        for offset in range(days):
            sleep_date = final_date - timedelta(days=(days - offset - 1))
            nightly = self._generate_night(sleep_date=sleep_date, log_id=100_000 + offset)
            records.append(nightly["session"])
            stages.extend(nightly["stages"])
            heart_rate_rows.extend(nightly["heart_rate"])
            hrv_rows.append(nightly["hrv"])
            spo2_rows.append(nightly["spo2"])

        return {
            "sleep_sessions": pd.DataFrame(records),
            "sleep_stages": pd.DataFrame(stages),
            "heart_rate": pd.DataFrame(heart_rate_rows),
            "hrv": pd.DataFrame(hrv_rows),
            "spo2": pd.DataFrame(spo2_rows),
        }

    def save(
        self,
        output_dir: str | Path,
        *,
        days: int = 14,
        end_date: date | None = None,
    ) -> dict[str, Path]:
        """Generate sample data and persist it as CSV files."""

        frames = self.generate(days=days, end_date=end_date)
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        written: dict[str, Path] = {}
        for name, frame in frames.items():
            path = target_dir / f"{name}.csv"
            frame.to_csv(path, index=False)
            written[name] = path

        manifest_path = target_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "seed": self.seed,
                    "days": days,
                    "end_date": (end_date or date.today()).isoformat(),
                    "files": {name: str(path.name) for name, path in written.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        written["manifest"] = manifest_path
        return written

    def _generate_night(self, *, sleep_date: date, log_id: int) -> dict[str, Any]:
        """Generate one night of session, stages, and biometrics."""

        bedtime_hour = int(self._rng.integers(21, 24))
        bedtime_minute = int(self._rng.choice([0, 10, 15, 20, 30, 40, 45, 50]))
        bedtime = datetime.combine(sleep_date, time(hour=bedtime_hour % 24, minute=bedtime_minute), tzinfo=timezone.utc)
        sleep_latency = int(self._rng.integers(5, 26))
        time_in_bed = int(self._rng.integers(390, 541))
        minutes_awake = int(self._rng.integers(15, 70))
        minutes_asleep = max(time_in_bed - minutes_awake, 240)
        awakenings_count = int(self._rng.integers(0, 5))
        efficiency = round((minutes_asleep / max(time_in_bed, 1)) * 100.0, 2)
        end_time = bedtime + timedelta(minutes=time_in_bed)

        deep_minutes = int(round(minutes_asleep * self._rng.uniform(0.16, 0.25)))
        rem_minutes = int(round(minutes_asleep * self._rng.uniform(0.18, 0.27)))
        wake_minutes = minutes_awake
        light_minutes = max(minutes_asleep - deep_minutes - rem_minutes, 0)

        session = {
            "logId": log_id,
            "dateOfSleep": sleep_date.isoformat(),
            "startTime": bedtime.isoformat(),
            "endTime": end_time.isoformat(),
            "minutesAsleep": minutes_asleep,
            "minutesAwake": minutes_awake,
            "timeInBed": time_in_bed,
            "minutesToFallAsleep": sleep_latency,
            "awakeningsCount": awakenings_count,
            "efficiency": efficiency,
            "levels_summary_deep_minutes": deep_minutes,
            "levels_summary_rem_minutes": rem_minutes,
            "levels_summary_light_minutes": light_minutes,
            "levels_summary_wake_minutes": wake_minutes,
        }

        stage_plan = [("light", light_minutes), ("deep", deep_minutes), ("rem", rem_minutes), ("wake", wake_minutes)]
        stage_rows: list[dict[str, Any]] = []
        stage_clock = bedtime + timedelta(minutes=sleep_latency)
        sequence = 0
        for stage_name, total_minutes in stage_plan:
            remaining = total_minutes
            if remaining <= 0:
                continue
            while remaining > 0:
                chunk = int(min(remaining, self._rng.choice([10, 15, 20, 30, 45])))
                started_at = stage_clock
                ended_at = started_at + timedelta(minutes=chunk)
                stage_rows.append(
                    {
                        "logId": log_id,
                        "dateOfSleep": sleep_date.isoformat(),
                        "stage_type": stage_name,
                        "started_at": started_at.isoformat(),
                        "ended_at": ended_at.isoformat(),
                        "seconds": chunk * 60,
                        "sequence_index": sequence,
                    }
                )
                stage_clock = ended_at
                remaining -= chunk
                sequence += 1

        heart_rate_rows: list[dict[str, Any]] = []
        hr_clock = bedtime
        while hr_clock <= end_time:
            hours_into_sleep = (hr_clock - bedtime).total_seconds() / 3600
            baseline_hr = 63 - min(hours_into_sleep * 1.8, 9)
            stage_modifier = float(self._rng.normal(0, 3))
            bpm = max(int(round(baseline_hr + stage_modifier)), 45)
            heart_rate_rows.append(
                {
                    "dateOfSleep": sleep_date.isoformat(),
                    "timestamp": hr_clock.isoformat(),
                    "heart_rate": bpm,
                }
            )
            hr_clock += timedelta(minutes=5)

        hrv = {
            "dateTime": sleep_date.isoformat(),
            "value_dailyRmssd": round(float(self._rng.uniform(32.0, 68.0)), 2),
        }
        spo2 = {
            "dateTime": sleep_date.isoformat(),
            "value_avg": round(float(self._rng.uniform(95.0, 98.8)), 2),
        }

        return {
            "session": session,
            "stages": stage_rows,
            "heart_rate": heart_rate_rows,
            "hrv": hrv,
            "spo2": spo2,
        }


def generate_sample_data(
    output_dir: str | Path = Path("data") / "sample",
    *,
    days: int = 14,
    seed: int = 42,
    end_date: date | None = None,
) -> dict[str, Path]:
    """Convenience wrapper to generate and save sample data."""

    return SyntheticSleepDataGenerator(seed=seed).save(output_dir, days=days, end_date=end_date)


__all__ = ["SyntheticSleepDataGenerator", "generate_sample_data"]
if __name__ == "__main__":
    print("Generating synthetic sleep data...")
    generate_sample_data()
    print("Sample data generated successfully.")