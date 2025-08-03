#!/usr/bin/env python3
"""
CBM training script optimized for Vertex AI.
Resumes from checkpoint and logs to MLflow (DagsHub).
"""

import sys
import os
import json
import argparse
from datetime import datetime
import uuid
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
import mlflow
import mlflow.pytorch
from tqdm import tqdm

from model.cbm_model import create_cbm_model
from model.cbm.concept_dataset import get_concept_data_loaders
from scripts.train_cbm_weighted import (
    ImprovedWeightedCBMTrainer,
    find_optimal_thresholds,
)


def setup_mlflow(experiment_name="cbm-vertex"):
    """Setup MLflow with DagsHub tracking"""
    # Set tracking URI from environment variable
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", "https://dagshub.com/kristina.kogan/art-dna.mlflow"
    )

    mlflow.set_tracking_uri(tracking_uri)
    print(f"📊 MLflow tracking URI: {tracking_uri}")

    # Set experiment
    experiment_name = f"{experiment_name}-{datetime.now().strftime('%Y%m%d')}"
    mlflow.set_experiment(experiment_name)

    # Start run
    run_name = f"vertex-t4-{uuid.uuid4().hex[:8]}"
    run = mlflow.start_run(run_name=run_name)

    print(f"🚀 Started MLflow run: {run_name}")
    return run


def download_training_data():
    """Download training data from GCS to local paths"""
    import subprocess

    print("📥 Downloading training data from GCS...")

    # Download CSV files
    csv_files = [
        ("gs://art-dna-ml-data/raw_data/final_df.csv", "raw_data/final_df.csv"),
        ("gs://art-dna-ml-data/raw_data/artists.csv", "raw_data/artists.csv"),
    ]

    for gcs_path, local_path in csv_files:
        print(f"   Downloading {gcs_path} -> {local_path}")
        subprocess.run(["gsutil", "cp", gcs_path, local_path], check=True)

    print("✅ Training data downloaded successfully")


def load_model_and_data(checkpoint_path, batch_size=16):
    """Load CBM model from checkpoint and prepare data loaders"""
    print("🧠 Loading CBM model and data...")

    # Download training data from GCS
    download_training_data()

    # Create model
    model = create_cbm_model()

    # Load checkpoint from GCS or local path
    if checkpoint_path.startswith("gs://"):
        # Download from GCS to temp location
        import subprocess

        temp_path = "/tmp/checkpoint.pth"
        subprocess.run(["gsutil", "cp", checkpoint_path, temp_path], check=True)
        checkpoint_path = temp_path

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"✅ Loaded checkpoint from {checkpoint_path}")
    print(f"   Trained for {checkpoint.get('epoch', 'unknown')} epochs")

    # Load data - now using local paths
    print("📁 Loading training data...")
    loaders = get_concept_data_loaders(batch_size=batch_size)

    print(f"✅ Data loaded - batch size: {batch_size}")

    return model, loaders, checkpoint


def save_checkpoint(model, optimizer, epoch, metrics, output_dir, is_best=False):
    """Save checkpoint to GCS"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }

    # Save locally first
    local_path = f"/tmp/checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, local_path)

    # Upload to GCS
    if output_dir.startswith("gs://"):
        import subprocess

        gcs_path = f"{output_dir}/checkpoint_epoch_{epoch}.pth"
        subprocess.run(["gsutil", "cp", local_path, gcs_path], check=True)

        if is_best:
            best_path = f"{output_dir}/best_checkpoint.pth"
            subprocess.run(["gsutil", "cp", local_path, best_path], check=True)
            print(f"💾 Best checkpoint saved to {best_path}")

        print(f"💾 Checkpoint saved to {gcs_path}")
    else:
        # Local save
        import shutil

        output_path = Path(output_dir) / f"checkpoint_epoch_{epoch}.pth"
        shutil.move(local_path, output_path)
        print(f"💾 Checkpoint saved to {output_path}")


def train_vertex_cbm(args):
    """Main training function for Vertex AI"""
    print("🎯 Starting Vertex AI CBM Training")
    print("=" * 50)

    # Set environment variable to indicate we're in Vertex AI
    import os

    os.environ["VERTEX_AI_TRAINING"] = "true"

    # Setup MLflow
    mlflow_run = setup_mlflow(args.experiment_name)

    # Log parameters
    mlflow.log_params(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "concept_weight": args.concept_weight,
            "checkpoint": args.checkpoint,
            "hardware": "T4-GPU" if torch.cuda.is_available() else "CPU",
        }
    )

    # Load model and data
    model, loaders, checkpoint = load_model_and_data(args.checkpoint, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"🔧 Using device: {device}")

    # Setup trainer (reuse existing weights logic)
    print("⚖️ Loading class weights...")

    # Try to load cached weights from GCS
    style_weights = None
    concept_weights = None

    # For now, compute weights from training data
    # TODO: Could optimize by caching these in GCS
    from model.pytorch_data_loader import calculate_pos_weights

    style_pos_weights = calculate_pos_weights("raw_data/final_df.csv")

    # Apply weight cap and move to device
    style_pos_weights = torch.clamp(style_pos_weights, max=args.weight_cap).to(device)

    # Calculate concept weights
    all_concept_labels = []
    for batch in tqdm(loaders["train"], desc="Computing concept weights"):
        _, _, concept_labels = batch
        all_concept_labels.append(concept_labels)
    all_concept_labels = torch.cat(all_concept_labels, dim=0)

    pos_counts = all_concept_labels.sum(dim=0)
    neg_counts = len(all_concept_labels) - pos_counts
    concept_pos_weights = neg_counts / (pos_counts + 1e-6)
    concept_pos_weights = torch.clamp(concept_pos_weights, max=args.weight_cap).to(
        device
    )

    print(f"✅ Weights computed:")
    print(
        f"   Style weights: {style_pos_weights.min():.2f} - {style_pos_weights.max():.2f}"
    )
    print(
        f"   Concept weights: {concept_pos_weights.min():.2f} - {concept_pos_weights.max():.2f}"
    )

    # Create trainer
    trainer = ImprovedWeightedCBMTrainer(
        model=model,
        style_pos_weights=style_pos_weights,
        concept_pos_weights=concept_pos_weights,
        concept_weight=args.concept_weight,
        style_weight=1.0,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_cap=args.weight_cap,
        device=device,
    )

    # Training loop
    best_f1 = 0.0
    best_epoch = 0
    start_epoch = checkpoint.get("epoch", 0)

    print(f"\n🏃 Starting training from epoch {start_epoch + 1}")
    print(f"Target: {args.epochs} epochs")

    for epoch in range(start_epoch + 1, start_epoch + 1 + args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}")
        print(f"{'='*60}")

        # Train
        train_loss = trainer.train_epoch(loaders["train"])

        # Evaluate
        val_metrics = trainer.evaluate(loaders["val"])
        val_f1 = val_metrics.get("style_f1_weighted", 0)

        print(f"📊 Epoch {epoch} Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_metrics['total_loss']:.4f}")
        print(f"   Val F1 (weighted): {val_f1:.4f}")

        # Log to MLflow
        mlflow.log_metrics(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["total_loss"],
                "val_f1_weighted": val_f1,
                "val_f1_micro": val_metrics.get("style_f1_micro", 0),
            },
            step=epoch,
        )

        # Check for improvement
        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
            best_epoch = epoch
            print(f"🏆 New best F1: {best_f1:.4f}")

        # Save checkpoint every 2 epochs
        if epoch % 2 == 0 or is_best:
            save_checkpoint(
                model, trainer.optimizer, epoch, val_metrics, args.output_dir, is_best
            )

        # Early stopping check
        if epoch >= 5 and val_f1 < 0.55:
            print(f"⏹️ Early stopping: F1 {val_f1:.3f} < 0.55 after epoch {epoch}")
            mlflow.set_tag("early_stopped", "true")
            mlflow.set_tag("early_stop_reason", f"F1 {val_f1:.3f} < 0.55")
            break

    # Final evaluation with optimal thresholds
    print("\n🔍 Final evaluation with optimal thresholds...")
    optimal_thresholds = find_optimal_thresholds(model, loaders["val"], device)
    final_metrics = trainer.evaluate(
        loaders["val"], optimal_thresholds=optimal_thresholds
    )
    final_f1 = final_metrics.get("style_f1_weighted", 0)

    print(f"🎯 Final Results:")
    print(f"   Best epoch: {best_epoch}")
    print(f"   Best F1: {best_f1:.4f}")
    print(f"   Final F1 (with optimal thresholds): {final_f1:.4f}")

    # Log final metrics
    mlflow.log_metrics(
        {
            "final_f1_weighted": final_f1,
            "best_f1_weighted": best_f1,
            "best_epoch": best_epoch,
            "total_epochs": epoch,
        }
    )

    # Save optimal thresholds
    thresholds_data = {
        "optimal_thresholds": optimal_thresholds.tolist(),
        "final_f1": final_f1,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
    }

    with open("/tmp/optimal_thresholds.json", "w") as f:
        json.dump(thresholds_data, f, indent=2)

    if args.output_dir.startswith("gs://"):
        import subprocess

        subprocess.run(
            [
                "gsutil",
                "cp",
                "/tmp/optimal_thresholds.json",
                f"{args.output_dir}/optimal_thresholds.json",
            ],
            check=True,
        )

    mlflow.end_run()
    print("✅ Training completed!")

    return final_f1


def main():
    parser = argparse.ArgumentParser(description="Train CBM on Vertex AI")

    # Model and data
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="gs://art-dna-ml-models/cbm/v1.0/cbm_weighted_best.pth",
        help="Checkpoint path (local or GCS)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gs://art-dna-ml-models/cbm/vertex-runs/",
        help="Output directory for checkpoints",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=12, help="Number of epochs to train"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--backbone-lr", type=float, default=1e-5, help="Learning rate for backbone"
    )
    parser.add_argument(
        "--head-lr", type=float, default=5e-5, help="Learning rate for heads"
    )
    parser.add_argument(
        "--concept-weight", type=float, default=0.3, help="Weight for concept loss"
    )
    parser.add_argument(
        "--weight-cap", type=float, default=10.0, help="Maximum class weight"
    )

    # MLflow
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="cbm-vertex",
        help="MLflow experiment name",
    )

    # Dry run for testing
    parser.add_argument(
        "--dry-run", action="store_true", help="Load model and data but don't train"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("🧪 DRY RUN MODE - Testing setup only")
        model, loaders, checkpoint = load_model_and_data(
            args.checkpoint, args.batch_size
        )
        print("✅ Dry run successful - all components loaded correctly")
        return

    # Run training
    final_f1 = train_vertex_cbm(args)

    # Set exit code based on success criteria
    if final_f1 >= 0.65:
        print(f"🎉 SUCCESS: Achieved F1 {final_f1:.4f} >= 0.65")
        sys.exit(0)
    else:
        print(f"⚠️ F1 {final_f1:.4f} < 0.65 target")
        sys.exit(1)


if __name__ == "__main__":
    main()
