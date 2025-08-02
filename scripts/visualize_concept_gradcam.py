"""
Generate Grad-CAM visualizations for trained concept classifiers.
Single-purpose script focused only on visualization.
"""

import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import Dataset


class SimpleConceptClassifier(nn.Module):
    """Simple binary concept classifier - must match training script."""

    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 1))

    def forward(self, x):
        return self.backbone(x)


class ConceptDataset(Dataset):
    """Dataset for loading concept samples."""

    def __init__(self, concept_scores, target_concept, transform=None):
        self.image_paths = []
        self.labels = []
        self.scores = []

        for path_with_run, scores in concept_scores.items():
            actual_path = path_with_run.replace("_run1", "").replace("_run2", "")

            if os.path.exists(actual_path):
                self.image_paths.append(actual_path)
                score = scores.get(target_concept, 0)
                self.labels.append(1 if score >= 0.7 else 0)
                self.scores.append(score)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], self.image_paths[idx], self.scores[idx]


def load_pilot_data():
    """Load concept scores from pilot results."""
    with open("model/cbm/pilot_results.json", "r") as f:
        pilot_data = json.load(f)

    # Combine both runs
    all_results = {}
    for img_path, scores in pilot_data["run1"].items():
        all_results[f"{img_path}_run1"] = scores
    for img_path, scores in pilot_data["run2"].items():
        all_results[f"{img_path}_run2"] = scores

    return all_results


def create_mock_trained_model(concept):
    """Create a mock trained model for testing (replace with actual loading later)."""
    print(f"🔄 Creating mock model for: {concept}")
    model = SimpleConceptClassifier()
    # For now, just return untrained model - in practice you'd load weights
    return model


def visualize_concept_gradcam(model, concept, concept_scores, n_samples=4):
    """Generate Grad-CAM visualizations for a concept."""
    print(f"\n🔍 Visualizing Grad-CAM for: {concept}")

    # Setup transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load dataset
    dataset = ConceptDataset(concept_scores, concept, transform)
    print(f"  Dataset size: {len(dataset)}")

    # Get samples to visualize
    pos_indices = [i for i in range(len(dataset)) if dataset[i][1] == 1]
    neg_indices = [i for i in range(len(dataset)) if dataset[i][1] == 0]

    # Select diverse samples
    test_indices = []
    if pos_indices:
        test_indices.extend(pos_indices[: n_samples // 2])
    if neg_indices:
        test_indices.extend(neg_indices[: n_samples // 2])
    test_indices = test_indices[:n_samples]

    if not test_indices:
        print(f"  ⚠️  No samples found for {concept}")
        return

    print(f"  Visualizing {len(test_indices)} samples")

    # Setup Grad-CAM (fixed API call)
    target_layers = [model.backbone.features[-1]]
    cam = GradCAM(
        model=model, target_layers=target_layers
    )  # Removed use_cuda parameter

    # Generate visualizations
    model.eval()
    for i, idx in enumerate(test_indices):
        image_tensor, label, image_path, raw_score = dataset[idx]

        # Model prediction
        with torch.no_grad():
            input_tensor = image_tensor.unsqueeze(0)
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()

        # Generate Grad-CAM
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Load original image
        original_image = Image.open(image_path).convert("RGB")
        original_image = original_image.resize((224, 224))
        original_np = np.array(original_image) / 255.0

        # Create visualization
        visualization = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(original_np)
        axes[0].set_title(f"Original Image\n{os.path.basename(image_path)}")
        axes[0].axis("off")

        # Grad-CAM heatmap only
        axes[1].imshow(grayscale_cam, cmap="jet")
        axes[1].set_title(f"Grad-CAM Heatmap\n{concept}")
        axes[1].axis("off")

        # Overlay
        axes[2].imshow(visualization)
        label_str = "POSITIVE" if label == 1 else "NEGATIVE"
        axes[2].set_title(
            f"Overlay\n{label_str} (pred: {prob:.3f})\nLLM score: {raw_score:.1f}"
        )
        axes[2].axis("off")

        plt.tight_layout()

        # Save
        output_path = f"model/cbm/gradcam_{concept}_sample_{i+1}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(
            f"    Sample {i+1}: {label_str} example (LLM: {raw_score:.1f}, pred: {prob:.3f})"
        )
        print(f"      Saved: {output_path}")


def main():
    print("🎨 CONCEPT GRAD-CAM VISUALIZATION")
    print("=" * 50)

    # Load concept data
    concept_scores = load_pilot_data()
    print(f"Loaded {len(concept_scores)} concept score samples")

    # Test concepts (manually specified for now)
    test_concepts = [
        "portraits",
        "religious_scenes",
        "calm_serene_mood",
        "vibrant_colors",
    ]

    print(f"\nVisualizing {len(test_concepts)} concepts:")
    for concept in test_concepts:
        print(f"  • {concept}")

    # Generate visualizations for each concept
    for concept in test_concepts:
        # Create/load model (mock for now)
        model = create_mock_trained_model(concept)

        # Generate visualizations
        try:
            visualize_concept_gradcam(model, concept, concept_scores, n_samples=4)
        except Exception as e:
            print(f"  ❌ Failed to visualize {concept}: {e}")
            continue

    print(f"\n✅ Visualization complete!")
    print(f"📁 Files saved to: model/cbm/gradcam_*.png")
    print(f"\n💡 Note: Using mock untrained models for demo")
    print(f"   In practice, load actual trained model weights")


if __name__ == "__main__":
    main()
