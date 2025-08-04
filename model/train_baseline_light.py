"""
Light training script for baseline validation on M1 Mac.
Optimized for speed to quickly validate approach.
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

from pytorch_data_loader import get_data_loaders, calculate_pos_weights


class SimpleArtStyleClassifier(nn.Module):
    """Simple EfficientNet-B3 baseline for multi-label art style classification."""

    def __init__(self, num_classes=18):
        super().__init__()

        # Use smaller EfficientNet-B0 for faster training
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True)

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

    return {
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
    }


def train_epoch_light(model, dataloader, criterion, optimizer, device, max_batches=50):
    """Train for one epoch with limited batches for speed."""
    model.train()
    running_loss = 0.0
    all_outputs = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training", total=min(len(dataloader), max_batches))
    for i, (images, labels) in enumerate(dataloader):
        if i >= max_batches:  # Limit batches for speed
            break

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

        pbar.update(1)
        pbar.set_postfix({"loss": loss.item()})

    pbar.close()

    # Calculate epoch metrics
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics(all_outputs, all_labels)

    avg_loss = running_loss / min(len(dataloader), max_batches)

    return avg_loss, metrics


def validate_light(model, dataloader, criterion, device, max_batches=20):
    """Validate the model with limited batches."""
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(
            dataloader, desc="Validation", total=min(len(dataloader), max_batches)
        )
        for i, (images, labels) in enumerate(dataloader):
            if i >= max_batches:
                break

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            all_outputs.append(outputs)
            all_labels.append(labels)

            pbar.update(1)
        pbar.close()

    # Calculate metrics
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics(all_outputs, all_labels)

    avg_loss = running_loss / min(len(dataloader), max_batches)

    return avg_loss, metrics


def train_baseline_light():
    """Light training for quick validation on M1 Mac."""

    # M1 Mac optimization: Use MPS if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using M1 Mac GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (will be slow)")

    # Light training parameters
    num_epochs = 3  # Just 3 epochs
    batch_size = 8  # Smaller batch for M1
    learning_rate = 5e-4  # Slightly higher LR for faster convergence
    image_size = 160  # Smaller images for speed

    # Create checkpoint directory
    checkpoint_dir = "model/checkpoints_light"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load data with optimizations
    print("\nLoading data (light mode)...")
    loaders = get_data_loaders(
        batch_size=batch_size,
        num_workers=0,  # 0 workers on M1 to avoid issues
        image_size=image_size,
        verbose=True,
    )

    # Create model
    print("\nCreating lightweight model (EfficientNet-B0)...")
    model = SimpleArtStyleClassifier(num_classes=18).to(device)

    # Loss and optimizer
    pos_weights = calculate_pos_weights().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training settings
    print(f"\n🚀 Light Training Settings:")
    print(f"  Device: {device}")
    print(f"  Model: EfficientNet-B0 (smaller)")
    print(f"  Image size: {image_size}x{image_size} (reduced)")
    print(f"  Batch size: {batch_size} (reduced)")
    print(f"  Epochs: {num_epochs}")
    print(f"  Max batches per epoch: 50 (train), 20 (val)")
    print("=" * 60)

    best_val_f1 = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train (limited batches)
        train_loss, train_metrics = train_epoch_light(
            model, loaders["train"], criterion, optimizer, device, max_batches=50
        )

        # Validate (limited batches)
        val_loss, val_metrics = validate_light(
            model, loaders["val"], criterion, device, max_batches=20
        )

        # Print metrics
        print(f"\nEpoch {epoch+1} Summary:")
        print(
            f"  Train Loss: {train_loss:.4f}, Train F1: {train_metrics['f1_macro']:.4f}"
        )
        print(f"  Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1_macro']:.4f}")

        # Save if best
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            checkpoint_path = os.path.join(checkpoint_dir, "best_model_light.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_f1": val_metrics["f1_macro"],
                },
                checkpoint_path,
            )
            print(f"  ✅ Model saved! (F1: {val_metrics['f1_macro']:.4f})")

        # Check if we're learning
        if val_metrics["f1_macro"] > 0.15:
            print(f"  📈 Model is learning! (F1 > 0.15)")

    print("\n" + "=" * 60)
    print("Light training complete!")
    print(f"Best Val F1: {best_val_f1:.4f}")
    print(f"\n💡 Next steps:")
    print("  1. Check if loss decreased and F1 increased")
    print("  2. Run Grad-CAM spot check: python scripts/spot_check_grad_cam_light.py")
    print("  3. If promising, train properly on Vertex AI")

    return model


if __name__ == "__main__":
    model = train_baseline_light()
