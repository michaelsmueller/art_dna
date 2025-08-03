#!/usr/bin/env python3
"""
CBM training for Vertex AI - optimized for maximum F1 performance.
Simplified version focusing on performance over concept diversity.
"""

import sys
import os
import argparse
import json
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
from model.cbm_model import create_cbm_model
from model.cbm.concept_dataset import get_concept_data_loaders
from scripts.train_cbm_weighted import (
    ImprovedWeightedCBMTrainer,
    find_optimal_thresholds,
)
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Train CBM on Vertex AI")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="cbm_weighted_best.pth",
        help="Starting checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vertex_models",
        help="Output directory for models",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--backbone-lr", type=float, default=5e-6, help="Learning rate for backbone"
    )
    parser.add_argument(
        "--head-lr", type=float, default=2e-5, help="Learning rate for heads"
    )
    parser.add_argument(
        "--weight-cap", type=float, default=10.0, help="Maximum class weight"
    )
    parser.add_argument(
        "--eval-every", type=int, default=1, help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--find-thresholds-every",
        type=int,
        default=3,
        help="Find optimal thresholds every N epochs",
    )
    parser.add_argument(
        "--distributed", action="store_true", help="Use distributed training"
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Setup device
    if args.distributed and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        device = torch.device("cuda")
        distributed = True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        distributed = False

    print(f"Training on device: {device}")

    # Load data
    print("Loading data...")
    loaders = get_concept_data_loaders(batch_size=args.batch_size)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Create model
    print("Creating model...")
    model = create_cbm_model()

    # Load checkpoint if provided
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    model = model.to(device)

    # Setup distributed training if needed
    if distributed:
        model = nn.DataParallel(model)

    # Load cached weights
    print("Loading class weights...")
    try:
        with open("style_weights_cache.json", "r") as f:
            style_weights_data = json.load(f)
            style_pos_weights = torch.tensor(style_weights_data["weights"]).to(device)
    except FileNotFoundError:
        print("Computing style weights...")
        from model.pytorch_data_loader import calculate_pos_weights

        style_pos_weights = calculate_pos_weights(
            loaders["train"], device=device, cap=args.weight_cap
        )
        # Cache for next time
        with open("style_weights_cache.json", "w") as f:
            json.dump({"weights": style_pos_weights.cpu().tolist()}, f)

    try:
        with open("concept_weights_cache.json", "r") as f:
            concept_weights_data = json.load(f)
            concept_pos_weights = torch.tensor(concept_weights_data["weights"]).to(
                device
            )
    except FileNotFoundError:
        print("Computing concept weights...")
        # Calculate from full training set
        all_concept_labels = []
        for batch in tqdm(train_loader, desc="Collecting concept labels"):
            _, _, concept_labels = batch
            all_concept_labels.append(concept_labels)
        all_concept_labels = torch.cat(all_concept_labels, dim=0)

        # Calculate weights
        pos_counts = all_concept_labels.sum(dim=0)
        neg_counts = len(all_concept_labels) - pos_counts
        concept_pos_weights = neg_counts / (pos_counts + 1e-6)
        concept_pos_weights = torch.clamp(concept_pos_weights, max=args.weight_cap).to(
            device
        )

        # Cache for next time
        with open("concept_weights_cache.json", "w") as f:
            json.dump({"weights": concept_pos_weights.cpu().tolist()}, f)

    print(
        f"Style weight range: {style_pos_weights.min():.2f} - {style_pos_weights.max():.2f}"
    )
    print(
        f"Concept weight range: {concept_pos_weights.min():.2f} - {concept_pos_weights.max():.2f}"
    )

    # Create trainer with aggressive settings for max F1
    trainer = ImprovedWeightedCBMTrainer(
        model=model,
        style_pos_weights=style_pos_weights,
        concept_pos_weights=concept_pos_weights,
        concept_weight=0.2,  # Reduce concept weight to prioritize style performance
        style_weight=1.0,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_cap=args.weight_cap,
        device=device,
    )

    # Load optimal thresholds if they exist
    optimal_thresholds = None
    if os.path.exists("optimal_thresholds.json"):
        print("Loading existing optimal thresholds...")
        with open("optimal_thresholds.json", "r") as f:
            optimal_thresholds = json.load(f)

    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0

    print("\nStarting training...")
    print(f"Epochs: {args.epochs}")
    print(f"Backbone LR: {args.backbone_lr}")
    print(f"Head LR: {args.head_lr}")
    print(f"Weight cap: {args.weight_cap}")

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")

        # Train
        train_loss = trainer.train_epoch(train_loader)

        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            val_metrics = trainer.evaluate(val_loader)

            # Find optimal thresholds periodically
            if (epoch + 1) % args.find_thresholds_every == 0:
                print("\nFinding optimal thresholds...")
                optimal_thresholds = find_optimal_thresholds(
                    model, val_loader, device, num_samples=2000
                )

                # Save thresholds
                threshold_path = output_dir / f"thresholds_epoch{epoch+1}.json"
                with open(threshold_path, "w") as f:
                    json.dump(optimal_thresholds, f, indent=2)

                # Re-evaluate with optimal thresholds
                val_metrics = trainer.evaluate(
                    val_loader, optimal_thresholds=optimal_thresholds
                )

            val_f1 = val_metrics.get(
                "style_f1_weighted", val_metrics.get("style_f1", 0)
            )

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['total_loss']:.4f}")
            print(
                f"Val Style F1 (weighted): {val_metrics.get('style_f1_weighted', 0):.4f}"
            )

            if optimal_thresholds:
                print(f"Val Style F1 (with optimal thresholds): {val_f1:.4f}")

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": (
                    model.module.state_dict() if distributed else model.state_dict()
                ),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "train_loss": train_loss,
                "val_metrics": val_metrics,
                "optimal_thresholds": optimal_thresholds,
            }

            checkpoint_path = output_dir / f"checkpoint_epoch{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_path)

            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1

                best_path = output_dir / "best_model.pth"
                torch.save(checkpoint, best_path)

                # Also save optimal thresholds
                if optimal_thresholds:
                    with open(output_dir / "best_thresholds.json", "w") as f:
                        json.dump(optimal_thresholds, f, indent=2)

                print(f"New best model! F1: {best_val_f1:.4f}")

    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
