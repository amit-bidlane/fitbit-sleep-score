"""PyTorch BiLSTM sleep stage classifier utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - depends on environment setup
    torch = None
    nn = None
    Dataset = object
    DataLoader = object
    Adam = None
    ReduceLROnPlateau = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


FEATURE_COLUMNS: tuple[str, ...] = (
    "heart_rate",
    "hrv",
    "movement",
    "spo2",
    "time_since_sleep_start",
    "rolling_hr_mean_5min",
)
CLASS_LABELS: dict[int, str] = {
    0: "awake",
    1: "light",
    2: "deep",
    3: "REM",
}


def _ensure_torch() -> None:
    """Raise a clear error when PyTorch is unavailable."""

    if torch is None:
        raise ImportError(
            "PyTorch is required for sleep classification. Install requirements.txt first."
        ) from TORCH_IMPORT_ERROR


def _to_numpy_array(data: Any, *, dtype: np.dtype | type[np.floating[Any]] = np.float32) -> np.ndarray:
    """Convert supported inputs to a NumPy array."""

    if isinstance(data, np.ndarray):
        return data.astype(dtype, copy=False)
    return np.asarray(data, dtype=dtype)


class SleepDataset(Dataset):
    """Sequence dataset for BiLSTM sleep stage classification."""

    sequence_length: int = 10
    num_features: int = 6

    def __init__(self, sequences: Any, labels: Any | None = None) -> None:
        _ensure_torch()
        sequence_array = _to_numpy_array(sequences)
        if sequence_array.ndim != 3:
            raise ValueError("Input sequences must be a 3D tensor: (batch_size, sequence_length, features).")
        if sequence_array.shape[1] != self.sequence_length:
            raise ValueError(f"sequence_length must be {self.sequence_length}.")
        if sequence_array.shape[2] != self.num_features:
            raise ValueError(f"features must be {self.num_features}.")

        self.sequences = torch.tensor(sequence_array, dtype=torch.float32)
        if labels is None:
            self.labels = None
        else:
            label_array = np.asarray(labels, dtype=np.int64)
            if label_array.ndim != 1:
                raise ValueError("Labels must be a 1D array-like collection.")
            if len(label_array) != len(sequence_array):
                raise ValueError("Sequences and labels must contain the same number of items.")
            self.labels = torch.tensor(label_array, dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""

        return int(self.sequences.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor] | Tensor:
        """Return a single sequence and, when available, its label."""

        sequence = self.sequences[index]
        if self.labels is None:
            return sequence
        return sequence, self.labels[index]


class SleepStageBiLSTM(nn.Module):
    """Bidirectional LSTM classifier for sleep stage prediction."""

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_classes: int = 4,
    ) -> None:
        _ensure_torch()
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.output_size = hidden_size * 2 if bidirectional else hidden_size

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.output_size, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return class logits for a batch of sequences."""

        outputs, _ = self.lstm(inputs)
        final_timestep = outputs[:, -1, :]
        final_timestep = self.dropout_layer(final_timestep)
        return self.classifier(final_timestep)


@dataclass(slots=True)
class SleepStageTrainer:
    """Trainer for the BiLSTM sleep stage classifier."""

    input_size: int = 6
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    learning_rate: float = 0.001
    epochs: int = 10
    batch_size: int = 32
    patience: int = 3
    device: str | None = None
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "sleep_classifier.pt"
    model: SleepStageBiLSTM = field(init=False)

    def __post_init__(self) -> None:
        _ensure_torch()
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SleepStageBiLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        ).to(self.device)

    @property
    def checkpoint_path(self) -> str:
        """Return the full checkpoint file path."""

        return os.path.join(self.checkpoint_dir, self.checkpoint_name)

    def fit(
        self,
        train_sequences: Any,
        train_labels: Any,
        *,
        val_sequences: Any | None = None,
        val_labels: Any | None = None,
    ) -> dict[str, list[float] | float | str]:
        """Train the classifier and save the best checkpoint."""

        train_dataset = SleepDataset(train_sequences, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        validation_loader = None
        if val_sequences is not None and val_labels is not None:
            validation_dataset = SleepDataset(val_sequences, val_labels)
            validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        history: dict[str, list[float] | float | str] = {
            "train_loss": [],
            "val_loss": [],
            "best_val_loss": float("inf"),
            "checkpoint_path": self.checkpoint_path,
        }

        best_metric = float("inf")
        epochs_without_improvement = 0

        for _epoch in range(self.epochs):
            train_loss = self._run_epoch(train_loader, optimizer, criterion)
            history["train_loss"].append(train_loss)

            monitored_loss = train_loss
            if validation_loader is not None:
                val_loss = self._evaluate(validation_loader, criterion)
                history["val_loss"].append(val_loss)
                monitored_loss = val_loss
            else:
                history["val_loss"].append(train_loss)

            scheduler.step(monitored_loss)

            if monitored_loss < best_metric:
                best_metric = monitored_loss
                history["best_val_loss"] = monitored_loss
                epochs_without_improvement = 0
                self.save_checkpoint(self.checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint(self.checkpoint_path)

        return history

    def _run_epoch(self, dataloader: DataLoader, optimizer: Adam, criterion: nn.Module) -> float:
        """Train for one epoch and return average loss."""

        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for sequences, labels in dataloader:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            logits = self.model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = int(labels.size(0))
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

        return total_loss / max(total_samples, 1)

    def _evaluate(self, dataloader: DataLoader, criterion: nn.Module) -> float:
        """Evaluate the model and return average loss."""

        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for sequences, labels in dataloader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(sequences)
                loss = criterion(logits, labels)

                batch_size = int(labels.size(0))
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size

        return total_loss / max(total_samples, 1)

    def save_checkpoint(self, checkpoint_path: str | None = None) -> str:
        """Persist the model configuration and weights."""

        path = checkpoint_path or self.checkpoint_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": {
                    "input_size": self.input_size,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "bidirectional": self.bidirectional,
                    "num_classes": 4,
                },
                "class_labels": CLASS_LABELS,
                "feature_columns": FEATURE_COLUMNS,
            },
            path,
        )
        return path

    def load_checkpoint(self, checkpoint_path: str | None = None) -> SleepStageBiLSTM:
        """Load weights from a saved checkpoint into the trainer model."""

        path = checkpoint_path or self.checkpoint_path
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        return self.model


@dataclass(slots=True)
class SleepClassifierInference:
    """Inference helpers for saved sleep stage classifiers."""

    checkpoint_path: str
    device: str | None = None
    model: SleepStageBiLSTM | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _ensure_torch()
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self) -> SleepStageBiLSTM:
        """Load the saved checkpoint and rebuild the model."""

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        config = checkpoint.get("model_config", {})
        self.model = SleepStageBiLSTM(
            input_size=config.get("input_size", 6),
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.3),
            bidirectional=config.get("bidirectional", True),
            num_classes=config.get("num_classes", 4),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        return self.model

    def predict(self, sequence: Any) -> int:
        """Return the predicted class index for one sequence."""

        probabilities = self.predict_proba(sequence)
        return int(np.argmax(probabilities))

    def predict_proba(self, sequence: Any) -> list[float]:
        """Return class probabilities for one sequence."""

        predictions = self.batch_predict([sequence], return_probabilities=True)
        return predictions[0]

    def batch_predict(
        self,
        sequences: Any,
        *,
        return_probabilities: bool = False,
    ) -> list[int] | list[list[float]]:
        """Predict class ids or probabilities for a batch of sequences."""

        model = self.model or self.load_model()
        dataset = SleepDataset(sequences)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_probabilities: list[list[float]] = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                logits = model(batch)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                all_probabilities.extend(probabilities.tolist())

        if return_probabilities:
            return all_probabilities
        return [int(np.argmax(row)) for row in all_probabilities]


def train_sleep_classifier(
    train_sequences: Any,
    train_labels: Any,
    *,
    val_sequences: Any | None = None,
    val_labels: Any | None = None,
    **trainer_kwargs: Any,
) -> tuple[SleepStageTrainer, dict[str, list[float] | float | str]]:
    """Train a sleep stage classifier with default settings."""

    trainer = SleepStageTrainer(**trainer_kwargs)
    history = trainer.fit(
        train_sequences=train_sequences,
        train_labels=train_labels,
        val_sequences=val_sequences,
        val_labels=val_labels,
    )
    return trainer, history


__all__ = [
    "CLASS_LABELS",
    "FEATURE_COLUMNS",
    "SleepClassifierInference",
    "SleepDataset",
    "SleepStageBiLSTM",
    "SleepStageTrainer",
    "train_sleep_classifier",
]
