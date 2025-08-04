#!/usr/bin/env python3
"""
CBM training with weighted loss - improved version with caching.
Key features:
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


class ImprovedWeightedCBMTrainer(CBMTrainer):
    """Improved CBM trainer with better handling of imbalanced data."""

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
    ):
        # Initialize parent with dummy learning rate
        super().__init__(
            model, concept_weight, style_weight, learning_rate=1e-4, device=device
        )

        # Cap weights
        style_pos_weights = torch.clamp(style_pos_weights, max=weight_cap)
        concept_pos_weights = torch.clamp(concept_pos_weights, max=weight_cap)

        # Override loss functions with weighted versions
        self.style_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=style_pos_weights.to(self.device)
        )
        self.concept_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=concept_pos_weights.to(self.device)
        )

        # Override optimizer with parameter groups
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        self.optimizer = torch.optim.Adam(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": head_lr},
            ]
        )

        print(f"✅ Improved weighted trainer initialized:")
        print(
            f"   Style weights: min={style_pos_weights.min():.2f}, max={style_pos_weights.max():.2f}"
        )
        print(
            f"   Concept weights: min={concept_pos_weights.min():.2f}, max={concept_pos_weights.max():.2f}"
        )
        print(f"   Concept loss weight: {concept_weight} (reduced to balance heads)")
        print(f"   Learning rates: backbone={backbone_lr}, heads={head_lr}")
        print(
            f"   Backbone params: {len(backbone_params)}, Head params: {len(head_params)}"
        )

    def train_epoch(self, train_loader, epoch):
        """Override to handle new metrics properly."""
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

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "Loss": f"{total_loss.item():.4f}",
                    "Style F1": f"{batch_metrics.get('style_f1_weighted', 0):.3f}",
                }
            )

        # Average over batches
        n_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches

        return epoch_losses, epoch_metrics

    def validate(self, val_loader):
        """Override to handle new metrics properly."""
        self.model.eval()

        val_losses = {"total": 0, "concept": 0, "style": 0}
        val_metrics = {
            "concept_accuracy": 0,
            "style_accuracy": 0,
            "concept_f1": 0,
            "style_f1": 0,
            "style_f1_weighted": 0,  # Add this!
            "style_f1_micro": 0,  # Add this!
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

                for key in val_metrics:
                    if key in metrics:
                        val_metrics[key] += metrics[key]

        # Average metrics
        n_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= n_batches
        for key in val_metrics:
            val_metrics[key] /= n_batches

        return val_losses, val_metrics


def load_or_calculate_concept_weights(train_loader, n_concepts, weight_cap=5.0):
    """Load cached concept weights or calculate them."""
    cache_path = Path("model/cbm/concept_weights_cache.json")

    if cache_path.exists():
        print("📊 Loading cached concept weights...")
        with open(cache_path, "r") as f:
            cache_data = json.load(f)

        concept_pos_weights = torch.tensor(cache_data["concept_pos_weights"])
        print(f"   Loaded weights for {len(concept_pos_weights)} concepts")
        print(
            f"   Concept weights: min={concept_pos_weights.min():.2f}, max={concept_pos_weights.max():.2f}"
        )
        return concept_pos_weights

    else:
        print(
            "📊 Calculating concept weights from full training set (this will be cached)..."
        )

        concept_pos_counts = torch.zeros(n_concepts)
        total_samples = 0

        with torch.no_grad():
            for images, _, concept_labels in tqdm(
                train_loader, desc="Scanning concepts"
            ):
                concept_pos_counts += (concept_labels > 0.5).float().sum(dim=0)
                total_samples += len(concept_labels)

        # Calculate rates and weights
        concept_pos_rates = concept_pos_counts / total_samples
        concept_neg_rates = 1 - concept_pos_rates
        concept_pos_weights = concept_neg_rates / (concept_pos_rates + 1e-6)
        concept_pos_weights = torch.clamp(concept_pos_weights, min=0.1, max=weight_cap)

        # Cache results
        cache_data = {
            "concept_pos_rates": concept_pos_rates.tolist(),
            "concept_pos_weights": concept_pos_weights.tolist(),
            "total_samples": total_samples,
            "weight_cap": weight_cap,
        }

        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        print(f"   Scanned {total_samples} samples")
        print(
            f"   Concept positive rates: min={concept_pos_rates.min():.3f}, max={concept_pos_rates.max():.3f}"
        )
        print(
            f"   Concept weights: min={concept_pos_weights.min():.2f}, max={concept_pos_weights.max():.2f}"
        )
        print(f"   ✅ Cached to {cache_path}")

        return concept_pos_weights


def find_optimal_thresholds(model, val_loader, device):
    """Find per-class optimal thresholds on validation set."""
    all_style_probs = []
    all_style_labels = []

    model.eval()
    with torch.no_grad():
        for images, style_labels, _ in val_loader:
            images = images.to(device)
            _, style_logits = model(images)
            style_probs = torch.sigmoid(style_logits)

            all_style_probs.append(style_probs.cpu())
            all_style_labels.append(style_labels)

    style_probs = torch.cat(all_style_probs, dim=0).numpy()
    style_labels = torch.cat(all_style_labels, dim=0).numpy()

    # Find best threshold per class
    n_classes = style_labels.shape[1]
    best_thresholds = np.zeros(n_classes)

    for i in range(n_classes):
        best_f1 = 0
        best_thresh = 0.5

        for thresh in np.arange(0.05, 0.55, 0.05):
            preds = (style_probs[:, i] >= thresh).astype(int)
            f1 = f1_score(style_labels[:, i], preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        best_thresholds[i] = best_thresh

    return best_thresholds


def main():
    print("🚀 CBM TRAINING WITH IMPROVED WEIGHTED LOSS")
    print("=" * 50)

    # Configuration
    config = {
        "batch_size": 16,
        "num_epochs": 15,
        "backbone_lr": 1e-5,
        "head_lr": 5e-5,
        "concept_weight": 0.3,
        "style_weight": 1.0,
        "weight_cap": 5.0,
        "patience": 3,
        "freeze_backbone_epochs": 0,
        "threshold_sweep_interval": 3,
    }

    print(f"📋 Configuration: {config}")

    # Load data
    print("\n📊 Loading data...")
    loaders = get_concept_data_loaders(batch_size=config["batch_size"], num_workers=0)

    # Calculate style weights from CSV
    style_pos_weights = calculate_pos_weights("raw_data/final_df.csv")
    print(f"\n📊 Style weights calculated from full dataset")

    # Load or calculate concept weights (with caching)
    n_concepts = 37
    concept_pos_weights = load_or_calculate_concept_weights(
        loaders["train"], n_concepts, config["weight_cap"]
    )

    # Create model
    print("\n🧠 Creating CBM model...")
    model = create_cbm_model()

    # Optionally freeze backbone initially
    if config["freeze_backbone_epochs"] > 0:
        print(
            f"❄️  Freezing backbone for first {config['freeze_backbone_epochs']} epochs"
        )
        for param in model.backbone.parameters():
            param.requires_grad = False

    # Create improved trainer
    trainer = ImprovedWeightedCBMTrainer(
        model=model,
        style_pos_weights=style_pos_weights,
        concept_pos_weights=concept_pos_weights,
        concept_weight=config["concept_weight"],
        style_weight=config["style_weight"],
        backbone_lr=config["backbone_lr"],
        head_lr=config["head_lr"],
        weight_cap=config["weight_cap"],
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
                "style_f1_weighted": train_metrics.get("style_f1_weighted", 0),
                "concept_accuracy": train_metrics["concept_accuracy"],
            }
        )
        history["val"].append(
            {
                "epoch": epoch + 1,
                "loss": val_losses["total"],
                "style_f1_weighted": val_metrics.get("style_f1_weighted", 0),
                "concept_accuracy": val_metrics["concept_accuracy"],
            }
        )

        # Early stopping based on LOSS (not F1)
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            epochs_without_improvement = 0

            print(f"\n💾 New best model! Val loss: {best_val_loss:.4f}")
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_f1_weighted": val_metrics.get("style_f1_weighted", 0),
                "optimal_thresholds": best_thresholds,
                "config": config,
            }
            torch.save(checkpoint, "model/cbm/models/cbm_weighted_best.pth")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= config["patience"]:
            print(
                f"\n⚠️  Early stopping after {config['patience']} epochs without loss improvement"
            )
            break

        # Check success gates at epoch 5
        if epoch == 4:
            print("\n📋 EPOCH 5 CHECKPOINT:")
            print(
                f"   ✓ Val weighted F1: {val_metrics.get('style_f1_weighted', 0):.3f} (target ≥ 0.35)"
            )
            print(
                f"   ✓ Concept accuracy: {val_metrics['concept_accuracy']:.3f} (target ≥ 0.70)"
            )
            print(f"   ✓ Best F1 with optimal thresholds: {best_val_f1_weighted:.3f}")

            if val_metrics.get("style_f1_weighted", 0) < 0.15:
                print("\n⚠️  Low F1 at epoch 5. Consider switching to focal loss.")

    # Final summary
    print("\n" + "=" * 50)
    print("📊 TRAINING SUMMARY")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Best Val F1 (with optimal thresholds): {best_val_f1_weighted:.3f}")
    print(f"   Target (VGG16 weighted F1): 0.431")

    # Save full results
    results = {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_f1_weighted": best_val_f1_weighted,
        "best_thresholds": (
            best_thresholds.tolist() if best_thresholds is not None else None
        ),
        "config": config,
    }

    with open("model/cbm/training_results_weighted.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Training complete! Results saved.")

    # Final gate check
    if best_val_f1_weighted >= 0.35 and history["val"][-1]["concept_accuracy"] >= 0.70:
        print("\n🎉 SUCCESS GATES PASSED! Ready for Vertex dry-run.")
    else:
        print("\n📝 Next steps: Consider focal loss or staged training.")

    return model, results


if __name__ == "__main__":
    main()
