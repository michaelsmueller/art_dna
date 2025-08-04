"""
Generate and save CBM heatmap visualizations.
Shows concept-level vs style-level attention.
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


class SimpleCBM(nn.Module):
    """Same CBM architecture for loading"""

    def __init__(self, n_concepts=37, n_styles=18):
        super().__init__()
        self.backbone = efficientnet_b0(pretrained=True)
        features = 1280
        self.backbone.classifier = nn.Identity()
        self.concept_head = nn.Linear(features, n_concepts)
        self.style_head = nn.Linear(n_concepts, n_styles)

    def forward(self, x, use_true_concepts=None):
        features = self.backbone(x)
        concept_logits = self.concept_head(features)

        if use_true_concepts is not None:
            concepts_for_style = use_true_concepts
        else:
            concepts_for_style = torch.sigmoid(concept_logits)

        style_logits = self.style_head(concepts_for_style)
        return concept_logits, style_logits


def load_cbm_model():
    """Load the trained CBM model"""
    model = SimpleCBM()
    model.load_state_dict(torch.load("model/cbm/simple_cbm_pilot.pth"))
    model.eval()
    return model


def visualize_cbm_comparison():
    """Generate side-by-side concept vs style heatmaps"""
    print("🎨 Generating CBM heatmap visualizations...")

    # Load model and data
    model = load_cbm_model()

    # Load first pilot sample
    with open("model/cbm/pilot_concepts_cbm.json", "r") as f:
        samples = json.load(f)

    test_sample = samples[0]
    image_path = test_sample["image_path"]

    # Load concepts and styles
    with open("model/cbm/final_concepts.json", "r") as f:
        concepts = json.load(f)["selected_concepts"]

    with open("model/class_names.txt", "r") as f:
        styles = [line.strip() for line in f]

    # Transform and predict
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    original_image = Image.open(image_path).convert("RGB")
    image_tensor = transform(original_image).unsqueeze(0)

    # Get predictions
    with torch.no_grad():
        concept_pred, style_pred = model(image_tensor)
        concept_probs = torch.sigmoid(concept_pred)
        style_probs = torch.sigmoid(style_pred)

    top_concept_idx = torch.argmax(concept_probs[0]).item()
    top_style_idx = torch.argmax(style_probs[0]).item()

    print(f"📸 Image: {os.path.basename(image_path)}")
    print(
        f"🎯 Top concept: {concepts[top_concept_idx]} ({concept_probs[0][top_concept_idx]:.3f})"
    )
    print(
        f"🎨 Top style: {styles[top_style_idx]} ({style_probs[0][top_style_idx]:.3f})"
    )

    # Generate heatmaps
    # 1. Style heatmap
    class StyleOnlyModel(nn.Module):
        def __init__(self, cbm_model):
            super().__init__()
            self.cbm = cbm_model

        def forward(self, x):
            _, style_logits = self.cbm(x)
            return style_logits

    style_model = StyleOnlyModel(model)
    style_cam = GradCAM(model=style_model, target_layers=[model.backbone.features[-1]])
    style_heatmap = style_cam(
        input_tensor=image_tensor, targets=[ClassifierOutputTarget(top_style_idx)]
    )

    # 2. Concept heatmap
    class ConceptOnlyModel(nn.Module):
        def __init__(self, cbm_model):
            super().__init__()
            self.cbm = cbm_model

        def forward(self, x):
            concept_logits, _ = self.cbm(x)
            return concept_logits

    concept_model = ConceptOnlyModel(model)
    concept_cam = GradCAM(
        model=concept_model, target_layers=[model.backbone.features[-1]]
    )
    concept_heatmap = concept_cam(
        input_tensor=image_tensor, targets=[ClassifierOutputTarget(top_concept_idx)]
    )

    # Create visualization
    original_resized = original_image.resize((224, 224))
    original_np = np.array(original_resized) / 255.0

    style_overlay = show_cam_on_image(original_np, style_heatmap[0], use_rgb=True)
    concept_overlay = show_cam_on_image(original_np, concept_heatmap[0], use_rgb=True)

    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: Style
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(style_heatmap[0], cmap="jet")
    axes[0, 1].set_title(f"Style Heatmap\n{styles[top_style_idx]}")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(style_overlay)
    axes[0, 2].set_title(f"Style Overlay\nPred: {style_probs[0][top_style_idx]:.3f}")
    axes[0, 2].axis("off")

    # Bottom row: Concept
    axes[1, 0].imshow(original_np)
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(concept_heatmap[0], cmap="jet")
    axes[1, 1].set_title(f"Concept Heatmap\n{concepts[top_concept_idx]}")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(concept_overlay)
    axes[1, 2].set_title(
        f"Concept Overlay\nPred: {concept_probs[0][top_concept_idx]:.3f}"
    )
    axes[1, 2].axis("off")

    plt.tight_layout()

    # Save
    output_path = "model/cbm/cbm_dual_gradcam_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ CBM heatmap comparison saved: {output_path}")

    # Calculate correlation
    correlation = np.corrcoef(style_heatmap[0].flatten(), concept_heatmap[0].flatten())[
        0, 1
    ]
    print(f"🔗 Style-Concept correlation: {correlation:.3f}")

    if abs(correlation) < 0.7:
        print("  ✅ Good! Style and concept focus on different regions")
    else:
        print("  ⚠️  High correlation - style and concept similar")


if __name__ == "__main__":
    visualize_cbm_comparison()
