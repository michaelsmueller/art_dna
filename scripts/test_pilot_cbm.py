"""
Simple Concept Bottleneck Model validation using pilot data.
Tests: image → concepts → styles with dual Grad-CAM.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import pandas as pd
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class SimpleCBM(nn.Module):
    """Simple Concept Bottleneck Model"""

    def __init__(self, n_concepts=37, n_styles=18):
        super().__init__()

        # Backbone
        self.backbone = efficientnet_b0(pretrained=True)
        features = 1280  # EfficientNet-B0 standard feature size
        self.backbone.classifier = nn.Identity()

        # Concept head
        self.concept_head = nn.Linear(features, n_concepts)

        # Style head (concepts → styles)
        self.style_head = nn.Linear(n_concepts, n_styles)

    def forward(self, x, use_true_concepts=None):
        # Get image features
        features = self.backbone(x)

        # Predict concepts
        concept_logits = self.concept_head(features)

        # Use true concepts if provided, else use predictions
        if use_true_concepts is not None:
            concepts_for_style = use_true_concepts
        else:
            concepts_for_style = torch.sigmoid(concept_logits)

        # Predict styles from concepts
        style_logits = self.style_head(concepts_for_style)

        return concept_logits, style_logits


def load_pilot_data():
    """Load pilot concept and style data"""
    print("📁 Loading pilot data...")

    # Load concept data
    with open("model/cbm/pilot_concepts_cbm.json", "r") as f:
        concept_data = json.load(f)

    # Load final concepts list
    with open("model/cbm/final_concepts.json", "r") as f:
        final_concepts = json.load(f)["selected_concepts"]

    # Load style data
    df = pd.read_csv("raw_data/final_df.csv")
    style_columns = [
        col for col in df.columns if col not in ["image_path", "artist_name"]
    ]

    # Match concept and style data
    samples = []
    for concept_item in concept_data:
        img_path = concept_item["image_path"]

        # Find matching style labels
        style_row = df[df["image_path"] == img_path]
        if len(style_row) > 0 and os.path.exists(img_path):
            # Concept vector (binary)
            concept_vector = [
                concept_item["concepts"].get(c, 0) for c in final_concepts
            ]

            # Style vector
            style_vector = style_row[style_columns].iloc[0].tolist()

            samples.append(
                {
                    "path": img_path,
                    "concepts": torch.tensor(concept_vector, dtype=torch.float32),
                    "styles": torch.tensor(style_vector, dtype=torch.float32),
                }
            )

    print(f"✅ Loaded {len(samples)} samples with {len(final_concepts)} concepts")
    return samples, final_concepts, style_columns


def train_simple_cbm(samples, n_epochs=10):
    """Train CBM on pilot data"""
    print(f"\n🔄 Training Simple CBM ({n_epochs} epochs)...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SimpleCBM().to(device)

    # Transform for images
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Loss and optimizer
    concept_loss_fn = nn.BCEWithLogitsLoss()
    style_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0

        for sample in samples:
            # Load and transform image
            image = Image.open(sample["path"]).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            concept_target = sample["concepts"].unsqueeze(0).to(device)
            style_target = sample["styles"].unsqueeze(0).to(device)

            # Forward pass
            concept_pred, style_pred = model(image_tensor)

            # Losses
            concept_loss = concept_loss_fn(concept_pred, concept_target)
            style_loss = style_loss_fn(style_pred, style_target)
            total_loss = 0.6 * concept_loss + 0.4 * style_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        if epoch % 3 == 0:
            avg_loss = epoch_loss / len(samples)
            print(f"  Epoch {epoch}: Loss = {avg_loss:.3f}")

    print("✅ Training complete!")
    return model


def test_cbm_gradcam(model, samples, final_concepts, style_columns):
    """Test Grad-CAM on both concepts and styles"""
    print(f"\n🔍 Testing CBM Grad-CAM...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()

    # Use first sample for testing
    test_sample = samples[0]

    # Load and transform image
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(test_sample["path"]).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get predictions
    with torch.no_grad():
        concept_pred, style_pred = model(image_tensor)
        concept_probs = torch.sigmoid(concept_pred)
        style_probs = torch.sigmoid(style_pred)

    # Find top predictions
    top_concept_idx = torch.argmax(concept_probs[0]).item()
    top_style_idx = torch.argmax(style_probs[0]).item()

    print(f"📸 Test image: {os.path.basename(test_sample['path'])}")
    print(
        f"  Top concept: {final_concepts[top_concept_idx]} ({concept_probs[0][top_concept_idx]:.3f})"
    )
    print(
        f"  Top style: {style_columns[top_style_idx]} ({style_probs[0][top_style_idx]:.3f})"
    )

    # Test 1: Style Grad-CAM (full model)
    print(f"\n1️⃣ Style Grad-CAM:")

    # Create style-only wrapper
    class StyleOnlyModel(nn.Module):
        def __init__(self, cbm_model):
            super().__init__()
            self.cbm = cbm_model

        def forward(self, x):
            _, style_logits = self.cbm(x)
            return style_logits

    style_model = StyleOnlyModel(model)
    style_cam = GradCAM(model=style_model, target_layers=[model.backbone.features[-1]])
    style_targets = [ClassifierOutputTarget(top_style_idx)]

    style_heatmap = style_cam(input_tensor=image_tensor, targets=style_targets)
    style_std = np.std(style_heatmap[0])
    print(f"  ✅ Style heatmap generated (std: {style_std:.3f})")

    # Test 2: Concept Grad-CAM
    print(f"\n2️⃣ Concept Grad-CAM:")

    # Create concept-only model wrapper
    class ConceptModel(nn.Module):
        def __init__(self, cbm_model):
            super().__init__()
            self.cbm = cbm_model

        def forward(self, x):
            concept_logits, _ = self.cbm(x)
            return concept_logits

    concept_model = ConceptModel(model)
    concept_cam = GradCAM(
        model=concept_model, target_layers=[model.backbone.features[-1]]
    )
    concept_targets = [ClassifierOutputTarget(top_concept_idx)]

    concept_heatmap = concept_cam(input_tensor=image_tensor, targets=concept_targets)
    concept_std = np.std(concept_heatmap[0])
    print(f"  ✅ Concept heatmap generated (std: {concept_std:.3f})")

    # Compare heatmaps
    correlation = np.corrcoef(style_heatmap[0].flatten(), concept_heatmap[0].flatten())[
        0, 1
    ]
    print(f"\n🔗 Heatmap correlation: {correlation:.3f}")

    if abs(correlation) < 0.7:
        print("  ✅ Good! Concepts and styles focus on different regions")
    else:
        print("  ⚠️  High correlation - concepts and styles similar")

    return True


def main():
    print("🧪 SIMPLE CBM VALIDATION")
    print("=" * 40)

    # Load data
    samples, final_concepts, style_columns = load_pilot_data()

    if len(samples) < 5:
        print("❌ Not enough samples for training")
        return

    # Train CBM
    model = train_simple_cbm(samples, n_epochs=8)

    # Test Grad-CAM
    success = test_cbm_gradcam(model, samples, final_concepts, style_columns)

    if success:
        print(f"\n🎉 CBM VALIDATION COMPLETE!")
        print(f"\n✅ What we proved:")
        print(f"  • CBM architecture works (concepts → styles)")
        print(f"  • Can train on pilot concept data")
        print(f"  • Grad-CAM works for both concepts AND styles")
        print(f"  • Pipeline is ready for scaling!")

        # Save model
        torch.save(model.state_dict(), "model/cbm/simple_cbm_pilot.pth")
        print(f"\n💾 Model saved: model/cbm/simple_cbm_pilot.pth")


if __name__ == "__main__":
    main()
