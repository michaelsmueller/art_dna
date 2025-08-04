#!/usr/bin/env python3
"""
Basic CBM training script with dual loss (concepts + styles).
Supports staged training and proper validation metrics.
"""

import sys
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from model.cbm_model import create_cbm_model
from model.cbm.concept_dataset import get_concept_data_loaders
from sklearn.metrics import f1_score
import numpy as np


class CBMTrainer:
    """Trainer for Concept Bottleneck Model with staged training support."""

    def __init__(
        self,
        model,
        concept_weight: float = 1.0,
        style_weight: float = 1.0,
        learning_rate: float = 1e-4,
        device: str = "auto",
    ):
        self.model = model
        self.concept_weight = concept_weight
        self.style_weight = style_weight

        # Device setup
        if device == "auto":
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Loss functions
        self.concept_loss_fn = nn.BCEWithLogitsLoss()
        self.style_loss_fn = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

        print(f"✅ CBM Trainer initialized on {self.device}")
        print(f"   Concept weight: {concept_weight}, Style weight: {style_weight}")
        print(f"   Learning rate: {learning_rate}")

    def compute_losses(
        self, concept_logits, style_logits, concept_labels, style_labels
    ):
        """Compute concept and style losses."""
        # Concept loss (always computed)
        concept_loss = self.concept_loss_fn(concept_logits, concept_labels)

        # Style loss (always computed)
        style_loss = self.style_loss_fn(style_logits, style_labels)

        # Combined loss
        total_loss = self.concept_weight * concept_loss + self.style_weight * style_loss

        return total_loss, concept_loss, style_loss

    def compute_metrics(
        self, concept_logits, style_logits, concept_labels, style_labels
    ):
        """Compute accuracy metrics."""
        with torch.no_grad():
            # Convert to probabilities
            concept_probs = torch.sigmoid(concept_logits)
            style_probs = torch.sigmoid(style_logits)

            # Convert to predictions (threshold = 0.5)
            concept_preds = (concept_probs > 0.5).float()
            style_preds = (style_probs > 0.5).float()

            # Move to CPU for sklearn
            concept_true = concept_labels.cpu().numpy()
            concept_pred = concept_preds.cpu().numpy()
            style_true = style_labels.cpu().numpy()
            style_pred = style_preds.cpu().numpy()

            # Compute metrics
            metrics = {}

            # Concept accuracy (element-wise)
            concept_acc = (concept_pred == concept_true).mean()
            metrics["concept_accuracy"] = concept_acc

            # Style accuracy (element-wise)
            style_acc = (style_pred == style_true).mean()
            metrics["style_accuracy"] = style_acc

            # F1 scores (macro-averaged)
            try:
                concept_f1 = f1_score(
                    concept_true, concept_pred, average="macro", zero_division=0
                )
                style_f1 = f1_score(
                    style_true, style_pred, average="macro", zero_division=0
                )
                metrics["concept_f1"] = concept_f1
                metrics["style_f1"] = style_f1
            except:
                metrics["concept_f1"] = 0.0
                metrics["style_f1"] = 0.0

            return metrics

    def train_epoch(self, train_loader, epoch_num):
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {"total": 0, "concept": 0, "style": 0}
        epoch_metrics = {
            "concept_accuracy": 0,
            "style_accuracy": 0,
            "concept_f1": 0,
            "style_f1": 0,
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch_num}")

        for batch_idx, (images, style_labels, concept_labels) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            style_labels = style_labels.to(self.device)
            concept_labels = concept_labels.to(self.device)

            # Forward pass
            concept_logits, style_logits = self.model(images)

            # Compute losses
            total_loss, concept_loss, style_loss = self.compute_losses(
                concept_logits, style_logits, concept_labels, style_labels
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Compute metrics
            metrics = self.compute_metrics(
                concept_logits, style_logits, concept_labels, style_labels
            )

            # Update running averages
            epoch_losses["total"] += total_loss.item()
            epoch_losses["concept"] += concept_loss.item()
            epoch_losses["style"] += style_loss.item()

            for key, value in metrics.items():
                epoch_metrics[key] += value

            # Update progress bar
            if batch_idx % 10 == 0:  # Update every 10 batches
                avg_loss = epoch_losses["total"] / (batch_idx + 1)
                avg_style_f1 = epoch_metrics["style_f1"] / (batch_idx + 1)
                pbar.set_postfix(
                    {"Loss": f"{avg_loss:.4f}", "Style F1": f"{avg_style_f1:.3f}"}
                )

        # Average metrics over epoch
        n_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches

        return epoch_losses, epoch_metrics

    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()

        val_losses = {"total": 0, "concept": 0, "style": 0}
        val_metrics = {
            "concept_accuracy": 0,
            "style_accuracy": 0,
            "concept_f1": 0,
            "style_f1": 0,
        }

        with torch.no_grad():
            for images, style_labels, concept_labels in tqdm(
                val_loader, desc="Validation"
            ):
                # Move to device
                images = images.to(self.device)
                style_labels = style_labels.to(self.device)
                concept_labels = concept_labels.to(self.device)

                # Forward pass
                concept_logits, style_logits = self.model(images)

                # Compute losses
                total_loss, concept_loss, style_loss = self.compute_losses(
                    concept_logits, style_logits, concept_labels, style_labels
                )

                # Compute metrics
                metrics = self.compute_metrics(
                    concept_logits, style_logits, concept_labels, style_labels
                )

                # Update running averages
                val_losses["total"] += total_loss.item()
                val_losses["concept"] += concept_loss.item()
                val_losses["style"] += style_loss.item()

                for key, value in metrics.items():
                    val_metrics[key] += value

        # Average metrics
        n_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= n_batches
        for key in val_metrics:
            val_metrics[key] /= n_batches

        return val_losses, val_metrics


def main():
    """Main training function."""
    print("🚀 CBM TRAINING - BASIC INFRASTRUCTURE TEST")
    print("=" * 50)

    # Training configuration
    config = {
        "batch_size": 16,  # Small for M1 Mac
        "num_epochs": 2,  # Just 2 epochs for testing
        "learning_rate": 1e-4,
        "concept_weight": 1.0,
        "style_weight": 1.0,
        "max_batches_per_epoch": 50,  # Limit for testing
    }

    print(f"📋 Configuration: {config}")

    # Create data loaders
    print("\n📊 Loading data...")
    loaders = get_concept_data_loaders(batch_size=config["batch_size"], num_workers=0)

    # Create model
    print("\n🧠 Creating CBM model...")
    model = create_cbm_model()

    # Create trainer
    trainer = CBMTrainer(
        model=model,
        concept_weight=config["concept_weight"],
        style_weight=config["style_weight"],
        learning_rate=config["learning_rate"],
    )

    # Training loop
    print(f"\n🏃 Starting training for {config['num_epochs']} epochs...")

    for epoch in range(config["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{config['num_epochs']} ---")

        # Limit batches for testing
        train_loader = loaders["train"]
        limited_train = []
        for i, batch in enumerate(train_loader):
            limited_train.append(batch)
            if i >= config["max_batches_per_epoch"] - 1:
                break

        # Train epoch
        train_losses, train_metrics = trainer.train_epoch(limited_train, epoch + 1)

        print(f"📈 Train Results:")
        print(
            f"   Loss: {train_losses['total']:.4f} (Concept: {train_losses['concept']:.4f}, Style: {train_losses['style']:.4f})"
        )
        print(
            f"   Style F1: {train_metrics['style_f1']:.3f}, Concept F1: {train_metrics['concept_f1']:.3f}"
        )
        print(
            f"   Style Acc: {train_metrics['style_accuracy']:.3f}, Concept Acc: {train_metrics['concept_accuracy']:.3f}"
        )

        # Validation (limited)
        val_loader = loaders["val"]
        limited_val = []
        for i, batch in enumerate(val_loader):
            limited_val.append(batch)
            if i >= 10 - 1:  # Just 10 validation batches
                break

        val_losses, val_metrics = trainer.validate(limited_val)

        print(f"📊 Val Results:")
        print(
            f"   Loss: {val_losses['total']:.4f} (Concept: {val_losses['concept']:.4f}, Style: {val_losses['style']:.4f})"
        )
        print(
            f"   Style F1: {val_metrics['style_f1']:.3f}, Concept F1: {val_metrics['concept_f1']:.3f}"
        )
        print(
            f"   Style Acc: {val_metrics['style_accuracy']:.3f}, Concept Acc: {val_metrics['concept_accuracy']:.3f}"
        )

    print("\n✅ CBM TRAINING INFRASTRUCTURE TEST COMPLETE!")
    print("\n🚀 Ready for next step: Staged training and MLflow integration")


if __name__ == "__main__":
    main()
