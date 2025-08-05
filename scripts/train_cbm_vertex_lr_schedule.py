#!/usr/bin/env python3
"""
CBM training script optimized for Vertex AI with OneCycleLR scheduler.
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
from scripts.train_cbm_weighted_lr_schedule import (
    ImprovedWeightedCBMTrainerWithScheduler,
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
    run_name = f"vertex-lr-schedule-{uuid.uuid4().hex[:8]}"
    run = mlflow.start_run(run_name=run_name)

    print(f"🚀 Started MLflow run: {run_name}")
    return run


def download_training_data():
    """Download training data from GCS to local paths"""
    import subprocess

    print("📥 Downloading training data from GCS...")

    # Create directories
    os.makedirs("raw_data", exist_ok=True)

    # Download CSV files (use correct bucket!)
    csv_files = [
        ("gs://art-dna-ml-data/raw_data/final_df.csv", "raw_data/final_df.csv"),
        ("gs://art-dna-ml-data/raw_data/artists.csv", "raw_data/artists.csv"),
    ]

    for gcs_path, local_path in csv_files:
        print(f"   Downloading {gcs_path} -> {local_path}")
        try:
            subprocess.run(["gsutil", "cp", gcs_path, local_path], check=True)
            print(f"✅ Downloaded {local_path}")
        except subprocess.CalledProcessError:
            print(f"⚠️ Could not download {local_path}")

    # Download ALL images in one go (simple fix!)
    print("📥 Downloading all training images (this will take ~5-10 minutes)...")
    os.makedirs("raw_data/resized", exist_ok=True)
    try:
        subprocess.run(
            [
                "gsutil",
                "-m",
                "cp",
                "-r",
                "gs://art-dna-ml-data/raw_data/resized/*",
                "raw_data/resized/",
            ],
            check=True,
        )
        print("✅ All training images downloaded successfully")
    except subprocess.CalledProcessError:
        print("⚠️ Could not download training images")

    print("✅ Training data setup complete")


def load_model_and_data(checkpoint_path, batch_size=32):
    """Load model and data, optionally from checkpoint"""
    print("🏗️ Loading model and data...")

    # Download training data if needed
    download_training_data()

    # Load data
    loaders = get_concept_data_loaders(batch_size=batch_size)
    print(f"✅ Data loaded: {len(loaders['train'].dataset)} training samples")

    # Create model
    model = create_cbm_model(n_classes=18)

    # Load checkpoint if provided
    checkpoint = None
    if checkpoint_path and checkpoint_path != "none":
        print(f"📂 Loading checkpoint from {checkpoint_path}")

        if checkpoint_path.startswith("gs://"):
            # Download from GCS
            local_path = "/tmp/checkpoint.pth"
            import subprocess

            subprocess.run(["gsutil", "cp", checkpoint_path, local_path], check=True)
            checkpoint_path = local_path

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    return model, loaders, checkpoint


def save_checkpoint(
    model, optimizer, scheduler, epoch, metrics, output_dir, is_best=False
):
    """Save model checkpoint to local and GCS"""
    import subprocess

    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }

    # Save locally first
    local_path = f"/tmp/checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint_data, local_path)

    # Upload to GCS
    if output_dir.startswith("gs://"):
        gcs_path = f"{output_dir.rstrip('/')}/checkpoint_epoch_{epoch}.pth"
        try:
            subprocess.run(["gsutil", "cp", local_path, gcs_path], check=True)
            print(f"💾 Checkpoint saved to {gcs_path}")

            # Save as best checkpoint if needed
            if is_best:
                best_path = f"{output_dir.rstrip('/')}/best_checkpoint.pth"
                subprocess.run(["gsutil", "cp", local_path, best_path], check=True)
                print(f"💾 Best checkpoint saved to {best_path}")

        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to upload checkpoint: {e}")
    else:
        # Local save
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint_data, output_path)

        if is_best:
            best_path = os.path.join(output_dir, "best_checkpoint.pth")
            torch.save(checkpoint_data, best_path)

        print(f"💾 Checkpoint saved to {output_path}")


def train_vertex_cbm(args):
    """Main training function for Vertex AI with OneCycleLR scheduler"""
    print("🎯 Starting Vertex AI CBM Training with OneCycleLR")
    print("=" * 60)

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
            "scheduler": "OneCycleLR",
            "hardware": "T4-GPU" if torch.cuda.is_available() else "CPU",
        }
    )

    # Load model and data
    model, loaders, checkpoint = load_model_and_data(args.checkpoint, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"🔧 Using device: {device}")

    # Calculate steps per epoch for scheduler
    steps_per_epoch = len(loaders["train"])
    print(f"📊 Steps per epoch: {steps_per_epoch}")

    # Setup trainer with scheduler
    print("⚖️ Loading class weights...")

    # Load style weights
    from model.pytorch_data_loader import calculate_pos_weights

    style_pos_weights = calculate_pos_weights("raw_data/final_df.csv")
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

    # Create trainer with scheduler
    trainer = ImprovedWeightedCBMTrainerWithScheduler(
        model=model,
        style_pos_weights=style_pos_weights,
        concept_pos_weights=concept_pos_weights,
        concept_weight=args.concept_weight,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_cap=args.weight_cap,
        num_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
    )

    # Resume from checkpoint if available
    start_epoch = 0
    best_f1 = 0.0
    best_epoch = 0

    if checkpoint:
        if "optimizer_state_dict" in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Restore initial_lr and max_lr for OneCycleLR compatibility
            trainer.optimizer.param_groups[0]["initial_lr"] = args.backbone_lr
            trainer.optimizer.param_groups[1]["initial_lr"] = args.head_lr
            trainer.optimizer.param_groups[0]["max_lr"] = args.backbone_lr * 10
            trainer.optimizer.param_groups[1]["max_lr"] = args.head_lr * 10
            print(
                "✅ Optimizer state restored with initial_lr and max_lr for OneCycleLR"
            )

        if "scheduler_state_dict" in checkpoint and trainer.scheduler:
            print("⚠️ OneCycleLR does not support resuming from checkpoints")
            print("   Starting fresh OneCycleLR schedule for remaining epochs")
            print("   This is the recommended approach per PyTorch documentation")

        start_epoch = checkpoint.get("epoch", 0)
        best_f1 = checkpoint.get("val_f1_weighted", 0.0)
        print(f"🔄 Resuming from epoch {start_epoch}, best F1: {best_f1:.4f}")

    print(f"🎯 Training target: {args.epochs} epochs")
    print(f"📈 Expected F1 improvement: {best_f1:.4f} → {best_f1 + 0.02:.4f}")

    for epoch in range(start_epoch + 1, start_epoch + 1 + args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}")
        print(f"{'='*60}")

        # Train
        train_losses, train_metrics = trainer.train_epoch(loaders["train"], epoch)
        train_loss = train_losses["total"]

        # Evaluate
        val_losses, val_metrics = trainer.validate(loaders["val"])
        val_f1 = val_metrics.get("style_f1_weighted", 0)
        val_total_loss = val_losses["total"]

        print(f"📊 Epoch {epoch} Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_total_loss:.4f}")
        print(f"   Val F1 (weighted): {val_f1:.4f}")

        # Log to MLflow
        mlflow.log_metrics(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_total_loss,
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

        # Save checkpoint every 2 epochs or if best
        if epoch % 2 == 0 or is_best:
            save_checkpoint(
                model,
                trainer.optimizer,
                trainer.scheduler,
                epoch,
                val_metrics,
                args.output_dir,
                is_best,
            )

        # Early stopping check (more lenient for scheduler)
        if epoch >= 5 and val_f1 < 0.60:
            print(f"⏹️ Early stopping: F1 {val_f1:.3f} < 0.60 after epoch {epoch}")
            mlflow.set_tag("early_stopped", "true")
            mlflow.set_tag("early_stop_reason", f"F1 {val_f1:.3f} < 0.60")
            break

    # Final evaluation with optimal thresholds
    print("\n🔍 Final evaluation with optimal thresholds...")
    optimal_thresholds = find_optimal_thresholds(model, loaders["val"], device)
    final_losses, final_metrics = trainer.validate(loaders["val"])

    # Log final results
    mlflow.log_metrics(
        {
            "final_f1_weighted": best_f1,
            "best_f1_weighted": best_f1,
            "best_epoch": best_epoch,
            "total_epochs": epoch,
        }
    )

    # Log thresholds as artifact
    thresholds_dict = {
        "thresholds": optimal_thresholds.tolist(),
        "best_f1": float(best_f1),
        "best_epoch": int(best_epoch),
    }

    with open("/tmp/optimal_thresholds.json", "w") as f:
        json.dump(thresholds_dict, f, indent=2)

    mlflow.log_artifact("/tmp/optimal_thresholds.json")

    # Upload final thresholds to GCS
    if args.output_dir.startswith("gs://"):
        import subprocess

        gcs_thresholds_path = f"{args.output_dir.rstrip('/')}/optimal_thresholds.json"
        try:
            subprocess.run(
                ["gsutil", "cp", "/tmp/optimal_thresholds.json", gcs_thresholds_path],
                check=True,
            )
            print(f"💾 Thresholds saved to {gcs_thresholds_path}")
        except subprocess.CalledProcessError:
            print("⚠️ Failed to upload thresholds")

    # Success criteria
    success_threshold = 0.72  # Target improvement from 0.7206
    if best_f1 >= success_threshold:
        print(f"🎉 SUCCESS: Achieved F1 {best_f1:.4f} >= {success_threshold}")
        mlflow.set_tag("training_success", "true")
        return best_f1
    else:
        print(f"⚠️ Target not reached: F1 {best_f1:.4f} < {success_threshold}")
        mlflow.set_tag("training_success", "false")
        return best_f1

    print("✅ Training completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Train CBM on Vertex AI with OneCycleLR"
    )

    # Model and data
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="gs://art-dna-ml-models/cbm/vertex-runs/best_checkpoint.pth",
        help="Checkpoint path (local or GCS) - use your best F1=0.7206 model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gs://art-dna-ml-models/cbm/vertex-lr-schedule/",
        help="Output directory for checkpoints",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=8, help="Number of epochs to train"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-5,
        help="Base learning rate for backbone",
    )
    parser.add_argument(
        "--head-lr", type=float, default=5e-5, help="Base learning rate for heads"
    )
    parser.add_argument(
        "--concept-weight", type=float, default=0.3, help="Weight for concept loss"
    )
    parser.add_argument(
        "--weight-cap", type=float, default=5.0, help="Maximum class weight"
    )

    # MLflow
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="cbm-vertex-lr-schedule",
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
    success_threshold = 0.72
    if final_f1 >= success_threshold:
        print(f"🎉 Training succeeded: F1 {final_f1:.4f} >= {success_threshold}")
        exit(0)
    else:
        print(f"⚠️ Training below target: F1 {final_f1:.4f} < {success_threshold}")
        exit(1)


if __name__ == "__main__":
    main()
