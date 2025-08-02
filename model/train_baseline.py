"""
Train baseline EfficientNet-B3 model for multi-label art style classification.
Goal: Achieve F1 ≥ 0.40 to validate approach before adding concepts.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
import json
from datetime import datetime

from pytorch_data_loader import get_data_loaders, calculate_pos_weights


class SimpleArtStyleClassifier(nn.Module):
    """Simple EfficientNet-B3 baseline for multi-label art style classification."""

    def __init__(self, num_classes=18):
        super().__init__()

        # Load pretrained EfficientNet-B3
        self.backbone = timm.create_model("efficientnet_b3", pretrained=True)

        # Replace classifier
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def calculate_metrics(outputs, labels, threshold=0.5):
    """Calculate F1, precision, recall for multi-label classification."""

    # Convert logits to predictions
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()

    # Move to CPU for sklearn
    preds_cpu = preds.cpu().numpy()
    labels_cpu = labels.cpu().numpy()

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_cpu, preds_cpu, average="macro", zero_division=0
    )

    # Also calculate per-class F1 for analysis
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        labels_cpu, preds_cpu, average=None, zero_division=0
    )

    return {
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_per_class": f1_per_class,
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_outputs = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        all_outputs.append(outputs.detach())
        all_labels.append(labels)

        pbar.set_postfix({"loss": loss.item()})

    # Calculate epoch metrics
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics(all_outputs, all_labels)

    avg_loss = running_loss / len(dataloader)

    return avg_loss, metrics


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            all_outputs.append(outputs)
            all_labels.append(labels)

    # Calculate metrics
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics(all_outputs, all_labels)

    avg_loss = running_loss / len(dataloader)

    return avg_loss, metrics


def train_baseline(
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-4,
    checkpoint_dir="model/checkpoints",
    device=None,
):
    """Train baseline model with monitoring."""

    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load data
    print("\nLoading data...")
    loaders = get_data_loaders(batch_size=batch_size, num_workers=4)

    # Create model
    print("\nCreating model...")
    model = SimpleArtStyleClassifier(num_classes=18).to(device)

    # Loss and optimizer
    pos_weights = calculate_pos_weights().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "best_val_f1": 0.0,
    }

    # Load genre names for reporting
    with open("model/class_names.txt", "r") as f:
        genre_names = [line.strip() for line in f if line.strip()]

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    print(f"Target: F1 ≥ 0.40")
    print("=" * 60)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, train_metrics = train_epoch(
            model, loaders["train"], criterion, optimizer, device
        )

        # Validate
        val_loss, val_metrics = validate(model, loaders["val"], criterion, device)

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_metrics["f1_macro"])
        history["val_f1"].append(val_metrics["f1_macro"])

        # Print metrics
        print(f"\nEpoch {epoch+1} Summary:")
        print(
            f"  Train Loss: {train_loss:.4f}, Train F1: {train_metrics['f1_macro']:.4f}"
        )
        print(f"  Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1_macro']:.4f}")

        # Save best model
        if val_metrics["f1_macro"] > history["best_val_f1"]:
            history["best_val_f1"] = val_metrics["f1_macro"]
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_metrics["f1_macro"],
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f"  ✅ New best model saved! (F1: {val_metrics['f1_macro']:.4f})")

        # Show per-class F1 every few epochs
        if (epoch + 1) % 3 == 0:
            print("\n  Per-class F1 scores:")
            for i, (genre, f1) in enumerate(
                zip(genre_names, val_metrics["f1_per_class"])
            ):
                print(f"    {genre:20s}: {f1:.3f}")

        # Early stopping check
        if val_metrics["f1_macro"] >= 0.40:
            print(
                f"\n🎯 Target F1 ≥ 0.40 achieved! Val F1: {val_metrics['f1_macro']:.4f}"
            )
            break

    # Save final model and history
    final_checkpoint = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        final_checkpoint,
    )

    # Save history as JSON
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Val F1: {history['best_val_f1']:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")

    return model, history


if __name__ == "__main__":
    # Train with small epochs for initial validation
    model, history = train_baseline(
        num_epochs=5,  # Start with just 5 epochs for validation
        batch_size=32,
        learning_rate=1e-4,
    )
