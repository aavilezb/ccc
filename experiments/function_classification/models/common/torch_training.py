"""Small supervised PyTorch training loop shared by sequence encoders."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .trace_dataset import SPLIT_IDS


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    history: list[dict]
    predictions: np.ndarray
    true_labels: np.ndarray
    test_indices: np.ndarray


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def train_classifier(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    *,
    mask: np.ndarray | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    seed: int = 2026,
) -> TrainingResult:
    train_indices = np.flatnonzero(splits == SPLIT_IDS["train"])
    validation_indices = np.flatnonzero(splits == SPLIT_IDS["validation"])
    test_indices = np.flatnonzero(splits == SPLIT_IDS["test"])
    if not len(train_indices) or not len(validation_indices) or not len(test_indices):
        raise ValueError("train, validation, and test splits must all be nonempty")

    tensors = [torch.from_numpy(x), torch.from_numpy(y)]
    if mask is not None:
        tensors.append(torch.from_numpy(mask))
    dataset = TensorDataset(*tensors)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices.tolist()),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        torch.utils.data.Subset(dataset, validation_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
    )

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    class_counts = np.bincount(y[train_indices], minlength=int(y.max()) + 1)
    if np.any(class_counts == 0):
        raise ValueError("every class must appear in the training split")
    class_weights = len(train_indices) / (len(class_counts) * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    history: list[dict] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in train_loader:
            batch = tuple(value.to(device) for value in batch)
            features, labels = batch[:2]
            batch_mask = batch[2] if len(batch) == 3 else None
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, batch_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(labels)
            train_count += len(labels)

        validation_loss, validation_accuracy = evaluate_loss(
            model, validation_loader, criterion, device
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss / train_count,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        print(
            f"epoch {epoch:3d}: train_loss={row['train_loss']:.5f} "
            f"val_loss={validation_loss:.5f} val_accuracy={validation_accuracy:.3f}"
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    predicted: list[np.ndarray] = []
    true: list[np.ndarray] = []
    with torch.no_grad():
        for batch in test_loader:
            batch = tuple(value.to(device) for value in batch)
            features, labels = batch[:2]
            batch_mask = batch[2] if len(batch) == 3 else None
            logits = model(features, batch_mask)
            predicted.append(logits.argmax(dim=1).cpu().numpy())
            true.append(labels.cpu().numpy())
    return TrainingResult(
        best_epoch=best_epoch,
        history=history,
        predictions=np.concatenate(predicted),
        true_labels=np.concatenate(true),
        test_indices=test_indices,
    )


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = tuple(value.to(device) for value in batch)
            features, labels = batch[:2]
            batch_mask = batch[2] if len(batch) == 3 else None
            logits = model(features, batch_mask)
            total_loss += float(criterion(logits, labels)) * len(labels)
            total_correct += int((logits.argmax(dim=1) == labels).sum())
            total += len(labels)
    return total_loss / total, total_correct / total
