#!/usr/bin/env python3
"""
Visualize Grad-CAM heatmaps for the weighted CBM model.
Shows what the model focuses on for both concept and style predictions.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import random

from model.cbm_model import ConceptBottleneckModel
from model.cbm.concept_dataset import get_concept_data_loaders


def load_trained_model(checkpoint_path="model/cbm/models/cbm_weighted_best.pth"):
    """Load the trained CBM model with optimal thresholds."""
    print("🧠 Loading trained CBM model...")

    model = ConceptBottleneckModel(
        n_concepts=37,
        n_classes=18,
        backbone_weights=None,  # Don't reload ImageNet weights
        freeze_backbone=False,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load optimal thresholds if available
    optimal_thresholds = checkpoint.get("optimal_thresholds", None)
    if optimal_thresholds is not None:
        print(f"✅ Loaded optimal thresholds: {optimal_thresholds[:5]}... (first 5)")

    return model, optimal_thresholds


def get_class_names():
    """Get style class names."""
    class_names = [
        "Abstractionism",
        "Art Nouveau",
        "Baroque",
        "Byzantine Art",
        "Cubism",
        "Expressionism",
        "Impressionism",
        "Mannerism",
        "Muralism",
        "Neoplasticism",
        "Pop Art",
        "Primitivism",
        "Realism",
        "Renaissance",
        "Romanticism",
        "Suprematism",
        "Surrealism",
        "Symbolism",
    ]
    return class_names


def get_concept_names():
    """Load concept names from the final concepts file."""
    with open("model/cbm/data/final_concepts.json", "r") as f:
        concepts = json.load(f)["selected_concepts"]
    return concepts


def create_gradcam_visualizations(
    model,
    images,
    labels,
    concept_labels,
    num_samples=5,
    save_dir="model/grad_cam_visual/weighted_cbm",
):
    """Generate Grad-CAM visualizations for both concepts and styles."""

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    class_names = get_class_names()
    concept_names = get_concept_names()

    # Target layers for Grad-CAM
    # For EfficientNet-B3, the last conv layer is in features[-1]
    target_layers = [model.backbone.features[-1]]

    # Create Grad-CAM objects
    concept_cam = GradCAM(model=model, target_layers=target_layers)
    style_cam = GradCAM(model=model, target_layers=target_layers)

    for idx in range(min(num_samples, len(images))):
        print(f"\n🎨 Processing sample {idx + 1}/{num_samples}...")

        # Get single image
        img = images[idx : idx + 1].to(device)
        label = labels[idx]
        concept_label = concept_labels[idx]

        # Get predictions
        with torch.no_grad():
            concept_logits, style_logits = model(img)
            style_probs = torch.sigmoid(style_logits).cpu()
            concept_probs = torch.sigmoid(concept_logits).cpu()

        # Find top predicted style and concept
        top_style_idx = style_probs[0].argmax().item()
        top_concept_idx = concept_probs[0].argmax().item()

        # Also find ground truth styles and concepts
        true_styles = [i for i, val in enumerate(label) if val > 0.5]
        true_concepts = [i for i, val in enumerate(concept_label) if val > 0.5]

        print(
            f"  Top predicted style: {class_names[top_style_idx]} ({style_probs[0, top_style_idx]:.3f})"
        )
        print(f"  True styles: {[class_names[i] for i in true_styles]}")
        print(
            f"  Top predicted concept: {concept_names[top_concept_idx]} ({concept_probs[0, top_concept_idx]:.3f})"
        )

        # Create visualization figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original image
        img_np = img[0].cpu().permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array(
            [0.485, 0.456, 0.406]
        )
        img_np = np.clip(img_np, 0, 1)

        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        # Style Grad-CAM for top predicted style
        def style_forward(x):
            concepts, styles = model(x)
            return styles

        style_targets = [ClassifierOutputTarget(top_style_idx)]
        style_grayscale_cam = style_cam(
            input_tensor=img, targets=style_targets, aug_smooth=True, eigen_smooth=True
        )

        style_cam_image = show_cam_on_image(
            img_np, style_grayscale_cam[0], use_rgb=True
        )
        axes[0, 1].imshow(style_cam_image)
        axes[0, 1].set_title(
            f"Style CAM: {class_names[top_style_idx]} ({style_probs[0, top_style_idx]:.3f})"
        )
        axes[0, 1].axis("off")

        # Style Grad-CAM for a true style (if exists and different)
        if true_styles and true_styles[0] != top_style_idx:
            true_style_idx = true_styles[0]
            style_targets = [ClassifierOutputTarget(true_style_idx)]
            style_grayscale_cam = style_cam(
                input_tensor=img,
                targets=style_targets,
                aug_smooth=True,
                eigen_smooth=True,
            )

            style_cam_image = show_cam_on_image(
                img_np, style_grayscale_cam[0], use_rgb=True
            )
            axes[0, 2].imshow(style_cam_image)
            axes[0, 2].set_title(f"Style CAM (True): {class_names[true_style_idx]}")
            axes[0, 2].axis("off")
        else:
            axes[0, 2].axis("off")

        # Concept Grad-CAM for top predicted concept
        # Need to temporarily modify model forward for concept CAM
        original_forward = model.forward

        def concept_forward(x):
            concepts, _ = original_forward(x)
            return concepts

        model.forward = concept_forward

        concept_targets = [ClassifierOutputTarget(top_concept_idx)]
        concept_grayscale_cam = concept_cam(
            input_tensor=img,
            targets=concept_targets,
            aug_smooth=True,
            eigen_smooth=True,
        )

        # Restore original forward
        model.forward = original_forward

        concept_cam_image = show_cam_on_image(
            img_np, concept_grayscale_cam[0], use_rgb=True
        )
        axes[1, 0].imshow(concept_cam_image)
        axes[1, 0].set_title(
            f"Concept CAM: {concept_names[top_concept_idx]} ({concept_probs[0, top_concept_idx]:.3f})"
        )
        axes[1, 0].axis("off")

        # Show two more interesting concepts
        interesting_concepts = [
            (i, prob.item())
            for i, prob in enumerate(concept_probs[0])
            if 0.3 < prob < 0.8
        ][:2]

        for ax_idx, (concept_idx, prob) in enumerate(interesting_concepts):
            if ax_idx >= 2:
                break

            # Use the same concept_forward function
            model.forward = concept_forward
            concept_targets = [ClassifierOutputTarget(concept_idx)]
            concept_grayscale_cam = concept_cam(
                input_tensor=img,
                targets=concept_targets,
                aug_smooth=True,
                eigen_smooth=True,
            )
            model.forward = original_forward

            concept_cam_image = show_cam_on_image(
                img_np, concept_grayscale_cam[0], use_rgb=True
            )
            axes[1, ax_idx + 1].imshow(concept_cam_image)
            axes[1, ax_idx + 1].set_title(
                f"Concept: {concept_names[concept_idx]} ({prob:.3f})"
            )
            axes[1, ax_idx + 1].axis("off")

        # Hide unused axes
        for i in range(len(interesting_concepts) + 1, 3):
            axes[1, i].axis("off")

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sample_{idx+1:02d}_gradcam_analysis.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Saved to {save_path}")

    return save_dir


def analyze_concept_style_alignment(model, test_loader, num_samples=10):
    """Analyze which concepts are most predictive of which styles."""
    print("\n🔍 Analyzing concept-style alignment...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    # Collect predictions
    all_concept_probs = []
    all_style_probs = []

    with torch.no_grad():
        for i, (images, style_labels, concept_labels) in enumerate(test_loader):
            if i >= num_samples:
                break

            images = images.to(device)
            concept_logits, style_logits = model(images)

            all_concept_probs.append(torch.sigmoid(concept_logits).cpu())
            all_style_probs.append(torch.sigmoid(style_logits).cpu())

    # Concatenate all predictions
    concept_probs = torch.cat(all_concept_probs, dim=0).numpy()
    style_probs = torch.cat(all_style_probs, dim=0).numpy()

    # Compute correlation matrix
    correlation_matrix = np.corrcoef(concept_probs.T, style_probs.T)
    n_concepts = concept_probs.shape[1]
    n_styles = style_probs.shape[1]

    # Extract concept-style correlations
    concept_style_corr = correlation_matrix[:n_concepts, n_concepts:]

    # Find strongest correlations
    class_names = get_class_names()
    concept_names = get_concept_names()

    print("\n📊 Strongest concept-style correlations:")
    for style_idx, style_name in enumerate(class_names):
        top_concepts = np.argsort(concept_style_corr[:, style_idx])[-3:][::-1]
        print(f"\n{style_name}:")
        for concept_idx in top_concepts:
            corr = concept_style_corr[concept_idx, style_idx]
            if corr > 0.1:  # Only show meaningful correlations
                print(f"  - {concept_names[concept_idx]}: {corr:.3f}")


def main():
    print("🎨 CBM GRAD-CAM VISUALIZATION")
    print("=" * 50)

    # Check if model exists
    checkpoint_path = "model/cbm/models/cbm_weighted_best.pth"
    if not Path(checkpoint_path).exists():
        print(f"❌ No checkpoint found at {checkpoint_path}")
        print("   Please complete training first!")
        return

    # Load model
    model, optimal_thresholds = load_trained_model(checkpoint_path)

    # Load test data
    print("\n📊 Loading test data...")
    loaders = get_concept_data_loaders(batch_size=8, num_workers=0)
    test_loader = loaders["test"]

    # Get a batch of test samples
    images, style_labels, concept_labels = next(iter(test_loader))

    # Generate visualizations
    print("\n🎨 Generating Grad-CAM visualizations...")
    save_dir = create_gradcam_visualizations(
        model, images, style_labels, concept_labels, num_samples=5
    )

    # Analyze concept-style alignment
    analyze_concept_style_alignment(model, test_loader, num_samples=20)

    print(f"\n✅ Visualizations saved to {save_dir}")
    print("\n📝 Summary:")
    print("  - Generated Grad-CAM heatmaps for styles and concepts")
    print("  - Analyzed concept-style correlations")
    print("  - Ready for qualitative evaluation!")


if __name__ == "__main__":
    main()
