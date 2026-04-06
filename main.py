"""CLI entry point for the Fitbit Sleep Score system."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.sample.generate_sample import generate_sample_data
from src.models.score_calculator import calculate_sleep_score
from src.models.sleep_classifier import train_sleep_classifier
from src.visualization.report_generator import SleepReportGenerator


def _load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""

    return pd.read_csv(Path(path))


def _print_json(payload: dict[str, Any]) -> None:
    """Pretty-print JSON to stdout."""

    print(json.dumps(payload, indent=2, default=str))


def run_api(args: argparse.Namespace) -> int:
    """Launch the FastAPI application with Uvicorn."""

    import uvicorn

    uvicorn.run(
        "src.api.routes:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=False,
    )
    return 0


def run_train(args: argparse.Namespace) -> int:
    """Train the sleep stage classifier on synthetic sequence data."""

    rng = np.random.default_rng(args.seed)
    train_sequences = rng.random((args.samples, 10, 6), dtype=np.float32)
    train_labels = rng.integers(0, 4, size=args.samples, dtype=np.int64)
    val_size = max(int(args.samples * 0.2), 1)
    val_sequences = rng.random((val_size, 10, 6), dtype=np.float32)
    val_labels = rng.integers(0, 4, size=val_size, dtype=np.int64)

    trainer, history = train_sleep_classifier(
        train_sequences=train_sequences,
        train_labels=train_labels,
        val_sequences=val_sequences,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
    )

    _print_json(
        {
            "mode": "train",
            "checkpoint_path": trainer.checkpoint_path,
            "best_val_loss": history["best_val_loss"],
            "epochs_recorded": len(history["train_loss"]),
        }
    )
    return 0


def run_analyze(args: argparse.Namespace) -> int:
    """Score nightly sleep features from a CSV file."""

    frame = _load_csv(args.input)
    scored = calculate_sleep_score(frame)
    output_path = Path(args.output) if args.output else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_path, index=False)

    preview = scored.head(args.limit).to_dict(orient="records")
    _print_json({"mode": "analyze", "rows": len(scored), "output": str(output_path) if output_path else None, "preview": preview})
    return 0


def run_report(args: argparse.Namespace) -> int:
    """Generate an HTML sleep report from a scores CSV and optional stages CSV."""

    scores = _load_csv(args.scores)
    summary = scores.iloc[[0]].copy()
    stages = _load_csv(args.stages) if args.stages else pd.DataFrame()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = SleepReportGenerator().generate_html_report(
        user_name=args.user_name,
        summary=summary,
        sleep_scores=scores,
        sleep_stages=stages,
        report_date=args.report_date or date.today().isoformat(),
        output_path=output_path,
    )
    _print_json({"mode": "report", "output": str(output_path), "html_length": len(html)})
    return 0


def run_trend(args: argparse.Namespace) -> int:
    """Summarize trend metrics from a scored CSV."""

    frame = _load_csv(args.input)
    date_column = next((name for name in ("score_date", "dateOfSleep", "date") if name in frame.columns), None)
    score_column = next((name for name in ("final_score", "overall_score", "score") if name in frame.columns), None)
    if date_column is None or score_column is None:
        raise ValueError("Trend input must include a date column and a score column.")

    trend_frame = frame.copy()
    trend_frame[date_column] = pd.to_datetime(trend_frame[date_column], errors="coerce")
    trend_frame[score_column] = pd.to_numeric(trend_frame[score_column], errors="coerce")
    trend_frame = trend_frame.dropna(subset=[date_column, score_column]).sort_values(date_column)
    trend_frame["rolling_average"] = trend_frame[score_column].rolling(window=min(args.window, len(trend_frame)), min_periods=1).mean()

    payload = {
        "mode": "trend",
        "rows": len(trend_frame),
        "latest_score": float(trend_frame.iloc[-1][score_column]) if not trend_frame.empty else None,
        "latest_rolling_average": float(trend_frame.iloc[-1]["rolling_average"]) if not trend_frame.empty else None,
        "preview": trend_frame.tail(args.limit).assign(
            **{date_column: trend_frame.tail(args.limit)[date_column].dt.strftime("%Y-%m-%d")}
        ).to_dict(orient="records"),
    }
    _print_json(payload)
    return 0


def run_migrate(args: argparse.Namespace) -> int:
    """Run Alembic migrations."""

    command = ["python", "-m", "alembic", args.action]
    if args.action == "upgrade":
        command.append(args.revision)
    elif args.action == "downgrade":
        command.append(args.revision)
    elif args.action == "revision":
        if args.message:
            command.extend(["-m", args.message])
        if args.autogenerate:
            command.append("--autogenerate")

    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def run_generate_sample(args: argparse.Namespace) -> int:
    """Generate synthetic sample sleep data files."""

    written = generate_sample_data(
        output_dir=args.output_dir,
        days=args.days,
        seed=args.seed,
    )
    _print_json({"mode": "generate-sample", "files": {key: str(value) for key, value in written.items()}})
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser and subcommands."""

    parser = argparse.ArgumentParser(description="Fitbit Sleep Score System CLI")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    api_parser = subparsers.add_parser("api", help="Run the FastAPI backend.")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", default=8000, type=int)
    api_parser.add_argument("--reload", action="store_true")
    api_parser.set_defaults(func=run_api)

    train_parser = subparsers.add_parser("train", help="Train the sleep stage classifier.")
    train_parser.add_argument("--samples", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--checkpoint-dir", default="checkpoints")
    train_parser.add_argument("--checkpoint-name", default="sleep_classifier.pt")
    train_parser.set_defaults(func=run_train)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze sleep features from a CSV file.")
    analyze_parser.add_argument("--input", required=True)
    analyze_parser.add_argument("--output")
    analyze_parser.add_argument("--limit", type=int, default=3)
    analyze_parser.set_defaults(func=run_analyze)

    report_parser = subparsers.add_parser("report", help="Generate an HTML sleep report.")
    report_parser.add_argument("--scores", required=True)
    report_parser.add_argument("--stages")
    report_parser.add_argument("--output", default=str(Path("data") / "sample" / "sleep-report.html"))
    report_parser.add_argument("--user-name", default="Fitbit User")
    report_parser.add_argument("--report-date")
    report_parser.set_defaults(func=run_report)

    trend_parser = subparsers.add_parser("trend", help="Summarize score trends from a CSV file.")
    trend_parser.add_argument("--input", required=True)
    trend_parser.add_argument("--window", type=int, default=7)
    trend_parser.add_argument("--limit", type=int, default=5)
    trend_parser.set_defaults(func=run_trend)

    migrate_parser = subparsers.add_parser("migrate", help="Run Alembic migrations.")
    migrate_parser.add_argument("action", choices=["upgrade", "downgrade", "revision", "current", "history"])
    migrate_parser.add_argument("--revision", default="head")
    migrate_parser.add_argument("--message")
    migrate_parser.add_argument("--autogenerate", action="store_true")
    migrate_parser.set_defaults(func=run_migrate)

    sample_parser = subparsers.add_parser("generate-sample", help="Create synthetic sample sleep data.")
    sample_parser.add_argument("--output-dir", default=str(Path("data") / "sample"))
    sample_parser.add_argument("--days", type=int, default=14)
    sample_parser.add_argument("--seed", type=int, default=42)
    sample_parser.set_defaults(func=run_generate_sample)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
