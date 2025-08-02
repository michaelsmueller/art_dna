"""Test a simple EfficientNet-B3 baseline model for multi-label classification."""

import sys

sys.path.append("model")

import torch
import torch.nn as nn
import timm
from pytorch_data_loader import get_data_loaders, calculate_pos_weights


class SimpleArtStyleClassifier(nn.Module):
    """Simple EfficientNet-B3 baseline for multi-label art style classification."""

    def __init__(self, num_classes=18):
        super().__init__()

        # Load pretrained EfficientNet-B3
        self.backbone = timm.create_model("efficientnet_b3", pretrained=True)

        # Get the number of features from the backbone
        # EfficientNet-B3 has 1536 features
        in_features = self.backbone.classifier.in_features

        # Replace classifier with our multi-label head
        self.backbone.classifier = nn.Linear(in_features, num_classes)

        print(f"Model created with {in_features} features → {num_classes} classes")

    def forward(self, x):
        # Forward pass (logits, not probabilities)
        return self.backbone(x)


def test_model():
    """Test the simple baseline model."""

    print("Testing EfficientNet-B3 baseline model...")
    print()

    # Create model
    model = SimpleArtStyleClassifier(num_classes=18)

    # Test with dummy input
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images

    print("Testing forward pass...")
    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {outputs.shape}")  # Should be [2, 18]
    print(f"  Output dtype: {outputs.dtype}")
    print(f"  Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    print()

    # Test with sigmoid (for inference)
    probabilities = torch.sigmoid(outputs)
    print(f"After sigmoid:")
    print(f"  Prob range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    print()

    # Test loss calculation
    print("Testing loss calculation...")
    pos_weights = calculate_pos_weights()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Dummy targets (multi-label)
    dummy_targets = torch.zeros(2, 18)
    dummy_targets[0, 6] = 1.0  # Impressionism
    dummy_targets[1, 16] = 1.0  # Surrealism

    loss = criterion(outputs, dummy_targets)
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Loss dtype: {loss.dtype}")
    print()

    # Test with real data loader (one batch)
    print("Testing with real data...")
    loaders = get_data_loaders(batch_size=4, num_workers=0, verbose=False)

    images, labels = next(iter(loaders["train"]))
    print(f"  Real batch shapes: {images.shape}, {labels.shape}")

    with torch.no_grad():
        real_outputs = model(images)
        real_loss = criterion(real_outputs, labels)

    print(f"  Real loss: {real_loss.item():.4f}")
    print()

    print("✅ Simple baseline model working correctly!")
    print("Ready for training loop!")


if __name__ == "__main__":
    test_model()
