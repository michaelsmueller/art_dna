"""Test one training step with EfficientNet-B3 baseline to verify training works."""

import sys

sys.path.append("model")

import torch
import torch.nn as nn
import torch.optim as optim
import timm
from pytorch_data_loader import get_data_loaders, calculate_pos_weights


class SimpleArtStyleClassifier(nn.Module):
    """Simple EfficientNet-B3 baseline for multi-label art style classification."""

    def __init__(self, num_classes=18):
        super().__init__()

        # Load pretrained EfficientNet-B3
        self.backbone = timm.create_model("efficientnet_b3", pretrained=True)

        # Get the number of features and replace classifier
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def test_training_step():
    """Test one training step to verify gradients and optimization work."""

    print("Testing one training step...")
    print()

    # Create model, optimizer, and loss
    model = SimpleArtStyleClassifier(num_classes=18)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Low learning rate
    pos_weights = calculate_pos_weights()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    print("✅ Model, optimizer, and loss created")
    print()

    # Get one batch of real data
    loaders = get_data_loaders(batch_size=4, num_workers=0, verbose=False)
    images, labels = next(iter(loaders["train"]))

    print(f"Training batch: {images.shape}, {labels.shape}")

    # Record initial state
    print("\nBefore training step:")

    # Get initial loss
    with torch.no_grad():
        initial_outputs = model(images)
        initial_loss = criterion(initial_outputs, labels)
        initial_probs = torch.sigmoid(initial_outputs)

    print(f"  Initial loss: {initial_loss.item():.4f}")
    print(
        f"  Initial prob range: [{initial_probs.min():.3f}, {initial_probs.max():.3f}]"
    )

    # Store initial weights for comparison
    initial_weight = model.backbone.classifier.weight.clone()

    # Training step
    print("\nPerforming training step...")

    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass
    loss.backward()

    # Check if gradients were computed
    has_gradients = any(p.grad is not None for p in model.parameters())
    print(f"  ✅ Gradients computed: {has_gradients}")

    # Check gradient norms
    total_grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm**0.5
    print(f"  Total gradient norm: {total_grad_norm:.4f}")

    # Optimizer step
    optimizer.step()

    print("\nAfter training step:")

    # Get final loss
    with torch.no_grad():
        final_outputs = model(images)
        final_loss = criterion(final_outputs, labels)
        final_probs = torch.sigmoid(final_outputs)

    print(f"  Final loss: {final_loss.item():.4f}")
    print(f"  Final prob range: [{final_probs.min():.3f}, {final_probs.max():.3f}]")

    # Check if weights changed
    final_weight = model.backbone.classifier.weight
    weight_change = (final_weight - initial_weight).abs().mean().item()
    print(f"  Weight change: {weight_change:.6f}")

    # Loss change
    loss_change = initial_loss.item() - final_loss.item()
    print(f"  Loss change: {loss_change:+.6f}")

    print()

    # Validation
    if has_gradients and weight_change > 1e-8 and total_grad_norm > 0:
        print("✅ Training step successful!")
        print("  - Gradients computed ✓")
        print("  - Weights updated ✓")
        print("  - Model is trainable ✓")
    else:
        print("❌ Training step issues detected")
        print(f"  - Gradients: {has_gradients}")
        print(f"  - Weight change: {weight_change}")
        print(f"  - Grad norm: {total_grad_norm}")

    print()
    print("Ready for full training loop!")


if __name__ == "__main__":
    test_training_step()
