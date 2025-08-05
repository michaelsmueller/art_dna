#!/usr/bin/env python3
"""
CBM training with weighted loss and OneCycleLR scheduler.
Key features:
- OneCycleLR learning rate scheduler for better convergence
- Cached concept weights for faster iteration
- Fixed metric handling for weighted F1
- Different LR for backbone vs heads
- Reduced concept weight (0.3) to balance head contributions
- Loss-based early stopping
- Per-class threshold optimization
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
from model.cbm_model import create_cbm_model
from model.cbm.concept_dataset import get_concept_data_loaders
from model.cbm.train_cbm import CBMTrainer
from model.pytorch_data_loader import calculate_pos_weights
import json
from sklearn.metrics import f1_score
from tqdm import tqdm
from pathlib import Path


class ImprovedWeightedCBMTrainerWithScheduler(CBMTrainer):
    """Improved CBM trainer with OneCycleLR scheduler."""

    def __init__(
        self,
        model,
        style_pos_weights,
        concept_pos_weights,
        concept_weight=0.3,
        style_weight=1.0,
        backbone_lr=1e-5,
        head_lr=5e-5,
        weight_cap=5.0,
        device="auto",
        num_epochs=10,
        steps_per_epoch=None,
    ):
        # Initialize parent with dummy learning rate
        super().__init__(
            model, concept_weight, style_weight, learning_rate=1e-4, device=device
        )

        # Store LR settings
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch

        # Apply weight capping
        style_pos_weights = torch.clamp(style_pos_weights, max=weight_cap).to(
            self.device
        )
        concept_pos_weights = torch.clamp(concept_pos_weights, max=weight_cap).to(
            self.device
        )

        # Create weighted losses
        self.style_criterion = nn.BCEWithLogitsLoss(pos_weight=style_pos_weights)
        self.concept_criterion = nn.BCEWithLogitsLoss(pos_weight=concept_pos_weights)

        # Create optimizer with separate learning rates
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if "backbone" in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": backbone_params,
                    "lr": backbone_lr,
                    "initial_lr": backbone_lr,
                },
                {"params": head_params, "lr": head_lr, "initial_lr": head_lr},
            ]
        )

        # Create OneCycleLR scheduler
        # Use the higher LR (head_lr) as max_lr since it's the dominant one
        if steps_per_epoch is not None:
            total_steps = num_epochs * steps_per_epoch
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[backbone_lr * 10, head_lr * 10],  # Peak at 10x base LR
                total_steps=total_steps,
                pct_start=0.1,  # 10% warmup
                anneal_strategy="cos",
                cycle_momentum=False,  # Adam doesn't use momentum
                div_factor=10.0,  # Start at max_lr/10
                final_div_factor=100.0,  # End at max_lr/100
            )
            print(f"✅ OneCycleLR scheduler initialized:")
            print(f"   Total steps: {total_steps}")
            print(
                f"   Max LRs: backbone={backbone_lr * 10:.2e}, heads={head_lr * 10:.2e}"
            )
            print(f"   Warmup steps: {int(total_steps * 0.1)}")
        else:
            self.scheduler = None
            print("⚠️  No scheduler initialized (steps_per_epoch not provided)")

        print(f"✅ Improved weighted trainer with scheduler initialized:")
        print(
            f"   Style weights: min={style_pos_weights.min():.2f}, max={style_pos_weights.max():.2f}"
        )
        print(
            f"   Concept weights: min={concept_pos_weights.min():.2f}, max={concept_pos_weights.max():.2f}"
        )
        print(f"   Concept loss weight: {concept_weight} (reduced to balance heads)")
        print(f"   Base learning rates: backbone={backbone_lr}, heads={head_lr}")
        print(
            f"   Backbone params: {len(backbone_params)}, Head params: {len(head_params)}"
        )

    def compute_metrics(
        self, concept_logits, style_logits, concept_labels, style_labels
    ):
        """Enhanced metrics including weighted and micro F1 scores."""
        # Get base metrics first
        metrics = super().compute_metrics(
            concept_logits, style_logits, concept_labels, style_labels
        )

        # Add missing weighted and micro F1 scores
        with torch.no_grad():
            # Convert to probabilities and predictions
            style_probs = torch.sigmoid(style_logits)
            style_preds = (style_probs > 0.5).float()

            # Move to CPU for sklearn
            style_true = style_labels.cpu().numpy()
            style_pred = style_preds.cpu().numpy()

            # Calculate additional F1 metrics
            try:
                style_f1_weighted = f1_score(
                    style_true, style_pred, average="weighted", zero_division=0
                )
                style_f1_micro = f1_score(
                    style_true, style_pred, average="micro", zero_division=0
                )

                metrics["style_f1_weighted"] = style_f1_weighted
                metrics["style_f1_micro"] = style_f1_micro

            except Exception as e:
                print(f"Warning: F1 calculation failed: {e}")
                metrics["style_f1_weighted"] = 0.0
                metrics["style_f1_micro"] = 0.0

        return metrics

    def train_epoch(self, train_loader, epoch):
        """Override to handle new metrics and scheduler."""
        self.model.train()
        epoch_losses = {"total": 0, "concept": 0, "style": 0}

        # Initialize ALL metrics we'll compute
        epoch_metrics = {
            "concept_accuracy": 0,
            "style_accuracy": 0,
            "concept_f1": 0,
            "style_f1": 0,
            "style_f1_weighted": 0,  # Add this!
            "style_f1_micro": 0,  # Add this!
        }

        # Track learning rates
        current_lrs = []
        if self.scheduler is not None:
            current_lrs = [group["lr"] for group in self.optimizer.param_groups]

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (images, style_labels, concept_labels) in enumerate(
            progress_bar
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

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Step the scheduler after each batch
            if self.scheduler is not None:
                self.scheduler.step()
                # Update current learning rates for display
                if batch_idx % 10 == 0:  # Update display every 10 batches
                    current_lrs = [group["lr"] for group in self.optimizer.param_groups]

            # Update losses
            epoch_losses["total"] += total_loss.item()
            epoch_losses["concept"] += concept_loss.item()
            epoch_losses["style"] += style_loss.item()

            # Compute and update metrics
            batch_metrics = self.compute_metrics(
                concept_logits, style_logits, concept_labels, style_labels
            )

            for key in epoch_metrics:
                if key in batch_metrics:
                    epoch_metrics[key] += batch_metrics[key]

            # Update progress bar with current LRs
            postfix_dict = {
                "Loss": f"{total_loss.item():.4f}",
                "Style F1": f"{batch_metrics.get('style_f1_weighted', 0):.3f}",
            }
            if current_lrs:
                postfix_dict["LR"] = f"{current_lrs[1]:.2e}"  # Show head LR

            progress_bar.set_postfix(postfix_dict)

        # Average over batches
        n_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches

        # Log final LRs for epoch
        if current_lrs:
            print(
                f"   Learning rates: backbone={current_lrs[0]:.2e}, heads={current_lrs[1]:.2e}"
            )

        return epoch_losses, epoch_metrics

    def validate(self, val_loader):
        """Validation without scheduler stepping."""
        self.model.eval()
        val_losses = {"total": 0, "concept": 0, "style": 0}

        # Initialize ALL metrics
        val_metrics = {
            "concept_accuracy": 0,
            "style_accuracy": 0,
            "concept_f1": 0,
            "style_f1": 0,
            "style_f1_weighted": 0,
            "style_f1_micro": 0,
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

                # Update losses
                val_losses["total"] += total_loss.item()
                val_losses["concept"] += concept_loss.item()
                val_losses["style"] += style_loss.item()

                # Compute metrics
                batch_metrics = self.compute_metrics(
                    concept_logits, style_logits, concept_labels, style_labels
                )

                for key in val_metrics:
                    if key in batch_metrics:
                        val_metrics[key] += batch_metrics[key]

        # Average over batches
        n_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= n_batches
        for key in val_metrics:
            val_metrics[key] /= n_batches

        return val_losses, val_metrics


def find_optimal_thresholds(model, val_loader, device, num_thresholds=20):
    """Find optimal per-class thresholds for multi-label classification."""
    model.eval()

    # Collect all predictions and labels
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, style_labels, _ in val_loader:
            images = images.to(device)
            _, style_logits = model(images)
            style_probs = torch.sigmoid(style_logits)

            all_probs.append(style_probs.cpu().numpy())
            all_labels.append(style_labels.numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    n_classes = all_probs.shape[1]
    optimal_thresholds = np.zeros(n_classes)

    # Find optimal threshold for each class
    for class_idx in range(n_classes):
        class_probs = all_probs[:, class_idx]
        class_labels = all_labels[:, class_idx]

        # Skip if no positive examples
        if class_labels.sum() == 0:
            optimal_thresholds[class_idx] = 0.5
            continue

        # Try different thresholds
        thresholds = np.linspace(0.1, 0.9, num_thresholds)
        best_f1 = 0
        best_threshold = 0.5

        for threshold in thresholds:
            preds = (class_probs >= threshold).astype(int)
            f1 = f1_score(class_labels, preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        optimal_thresholds[class_idx] = best_threshold

    return optimal_thresholds


def main():
    # Configuration - FAST LOCAL TEST
    config = {
        # Model
        "num_concepts": 37,
        "num_styles": 18,
        "pretrained": True,
        "freeze_backbone_epochs": 0,  # Train everything from start
        # Training - REDUCED FOR SPEED
        "batch_size": 64,  # Larger batches = fewer steps
        "num_epochs": 3,  # Just 3 epochs for quick test
        "backbone_lr": 1e-5,
        "head_lr": 5e-5,
        # Loss weights
        "concept_weight": 0.3,  # Reduced to balance dual heads
        "style_weight": 1.0,
        "weight_cap": 5.0,  # Cap extreme weights
        # Validation - REDUCED FOR SPEED
        "early_stopping_patience": 2,  # Stop after 2 epochs without improvement
        "threshold_sweep_interval": 10,  # Skip threshold optimization for speed
        # Paths
        "checkpoint_dir": "model/cbm/models",
        "final_model_name": "cbm_weighted_lr_schedule_test.pth",
    }

    # Create checkpoint directory
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CBM Training with Weighted Loss and OneCycleLR Scheduler")
    print("=" * 60)

    # Load data
    print("\n📊 Loading data...")
    loaders = get_concept_data_loaders(batch_size=config["batch_size"])
    print(f"✅ Data loaded: {len(loaders['train'].dataset)} training samples")

    # Calculate steps per epoch for scheduler
    steps_per_epoch = len(loaders["train"])
    print(f"   Steps per epoch: {steps_per_epoch}")

    # Create model
    print("\n🏗️ Creating model...")
    model = create_cbm_model(
        n_classes=config["num_styles"],
        backbone_weights="IMAGENET1K_V1" if config["pretrained"] else None,
    )

    # Calculate positive weights
    print("\n⚖️ Calculating class weights...")

    # Load cached concept weights
    cache_file = Path(__file__).parent.parent / "model/cbm/concept_weights_cache.json"
    if cache_file.exists():
        print("📂 Loading cached concept weights...")
        with open(cache_file, "r") as f:
            cache = json.load(f)
            concept_pos_weights = torch.tensor(cache["concept_pos_weights"])
        print(f"✅ Loaded weights for {len(concept_pos_weights)} concepts")
    else:
        print("❌ No cached concept weights found! Run cache_concept_weights.py first")
        return

    # Calculate style weights (these are fast)
    style_pos_weights = calculate_pos_weights("raw_data/final_df.csv")

    # Create improved trainer with scheduler
    trainer = ImprovedWeightedCBMTrainerWithScheduler(
        model=model,
        style_pos_weights=style_pos_weights,
        concept_pos_weights=concept_pos_weights,
        concept_weight=config["concept_weight"],
        style_weight=config["style_weight"],
        backbone_lr=config["backbone_lr"],
        head_lr=config["head_lr"],
        weight_cap=config["weight_cap"],
        num_epochs=config["num_epochs"],
        steps_per_epoch=steps_per_epoch,
    )

    print(f"\n🏃 Starting training for {config['num_epochs']} epochs...")

    best_val_loss = float("inf")
    best_val_f1_weighted = 0.0
    epochs_without_improvement = 0
    history = {"train": [], "val": []}
    best_thresholds = None

    for epoch in range(config["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{config['num_epochs']} ---")

        # Unfreeze backbone after specified epochs
        if (
            epoch == config["freeze_backbone_epochs"]
            and config["freeze_backbone_epochs"] > 0
        ):
            print("🔥 Unfreezing backbone")
            for param in model.backbone.parameters():
                param.requires_grad = True

        # Train
        train_losses, train_metrics = trainer.train_epoch(loaders["train"], epoch + 1)

        print(f"📈 Train Results:")
        print(
            f"   Loss: {train_losses['total']:.4f} (C: {train_losses['concept']:.4f}, S: {train_losses['style']:.4f})"
        )
        print(
            f"   Style F1: macro={train_metrics['style_f1']:.3f}, weighted={train_metrics.get('style_f1_weighted', 0):.3f}, micro={train_metrics.get('style_f1_micro', 0):.3f}"
        )
        print(
            f"   Accuracy: style={train_metrics['style_accuracy']:.3f}, concept={train_metrics['concept_accuracy']:.3f}"
        )

        # Validation
        val_losses, val_metrics = trainer.validate(loaders["val"])

        print(f"📊 Val Results:")
        print(
            f"   Loss: {val_losses['total']:.4f} (C: {val_losses['concept']:.4f}, S: {val_losses['style']:.4f})"
        )
        print(
            f"   Style F1: macro={val_metrics['style_f1']:.3f}, weighted={val_metrics.get('style_f1_weighted', 0):.3f}, micro={val_metrics.get('style_f1_micro', 0):.3f}"
        )
        print(
            f"   Accuracy: style={val_metrics['style_accuracy']:.3f}, concept={val_metrics['concept_accuracy']:.3f}"
        )

        # Periodic threshold optimization
        if (epoch + 1) % config["threshold_sweep_interval"] == 0:
            print("\n🔍 Finding optimal thresholds...")
            optimal_thresholds = find_optimal_thresholds(
                model, loaders["val"], trainer.device
            )

            # Compute F1 with optimal thresholds
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, style_labels, _ in loaders["val"]:
                    images = images.to(trainer.device)
                    _, style_logits = model(images)
                    style_probs = torch.sigmoid(style_logits).cpu().numpy()

                    # Apply per-class thresholds
                    preds = (style_probs >= optimal_thresholds).astype(int)
                    all_preds.append(preds)
                    all_labels.append(style_labels.numpy())

            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)

            optimal_f1_weighted = f1_score(
                all_labels, all_preds, average="weighted", zero_division=0
            )
            print(f"   F1 with optimal thresholds: {optimal_f1_weighted:.3f}")
            print(f"   Thresholds: {optimal_thresholds[:5]}... (first 5 classes)")

            if optimal_f1_weighted > best_val_f1_weighted:
                best_val_f1_weighted = optimal_f1_weighted
                best_thresholds = optimal_thresholds

        # Track history
        history["train"].append(
            {
                "epoch": epoch + 1,
                "loss": train_losses["total"],
                "f1_weighted": train_metrics.get("style_f1_weighted", 0),
            }
        )
        history["val"].append(
            {
                "epoch": epoch + 1,
                "loss": val_losses["total"],
                "f1_weighted": val_metrics.get("style_f1_weighted", 0),
            }
        )

        # Check for improvement (using weighted F1)
        current_val_f1 = val_metrics.get("style_f1_weighted", 0)

        if current_val_f1 > best_val_f1_weighted:
            best_val_f1_weighted = current_val_f1
            epochs_without_improvement = 0

            # Save best model
            checkpoint_path = (
                Path(config["checkpoint_dir"]) / config["final_model_name"]
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "scheduler_state_dict": (
                        trainer.scheduler.state_dict() if trainer.scheduler else None
                    ),
                    "train_loss": train_losses["total"],
                    "val_loss": val_losses["total"],
                    "val_f1_weighted": best_val_f1_weighted,
                    "config": config,
                    "history": history,
                    "thresholds": best_thresholds,
                },
                checkpoint_path,
            )
            print(f"💾 Saved best model (F1: {best_val_f1_weighted:.3f})")
        else:
            epochs_without_improvement += 1
            print(
                f"📉 No improvement for {epochs_without_improvement} epochs (best F1: {best_val_f1_weighted:.3f})"
            )

        # Early stopping
        if epochs_without_improvement >= config["early_stopping_patience"]:
            print(
                f"\n🛑 Early stopping triggered after {epoch + 1} epochs (no improvement for {config['early_stopping_patience']} epochs)"
            )
            break

    # Final threshold optimization
    print("\n🔍 Final threshold optimization...")
    final_thresholds = find_optimal_thresholds(model, loaders["val"], trainer.device)

    # Save final model with thresholds
    final_checkpoint_path = Path(config["checkpoint_dir"]) / config["final_model_name"]
    checkpoint = torch.load(final_checkpoint_path)
    checkpoint["final_thresholds"] = final_thresholds.tolist()
    torch.save(checkpoint, final_checkpoint_path)

    # Save thresholds separately
    threshold_path = Path(config["checkpoint_dir"]) / "val_thresholds_lr_schedule.json"
    with open(threshold_path, "w") as f:
        json.dump(
            {
                "thresholds": final_thresholds.tolist(),
                "val_f1_weighted": float(best_val_f1_weighted),
                "training_config": config,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation F1 (weighted): {best_val_f1_weighted:.3f}")
    print(f"Model saved to: {final_checkpoint_path}")
    print(f"Thresholds saved to: {threshold_path}")


if __name__ == "__main__":
    main()
