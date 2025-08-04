#!/usr/bin/env python3
"""
Collect CBM v1.1 predictions and save to files for separate analysis.
"""

import sys
import torch
import numpy as np
import json
import os

sys.path.append(".")

from model.cbm_model import ConceptBottleneckModel
from model.cbm.concept_dataset import get_concept_data_loaders
from scripts.train_cbm_weighted import find_optimal_thresholds


def collect_predictions():
    """Collect all predictions and save to files."""

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Load model
    print("📦 Loading v1.1 model...")
    model = ConceptBottleneckModel(n_concepts=36, n_classes=18).to(device)

    checkpoint = torch.load(
        "model/cbm/models/cbm_decorrelated_best.pth",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load data
    print("📊 Loading data...")
    loaders = get_concept_data_loaders(
        concept_json_path="model/cbm/data/merged_concepts/full_concepts_complete.json",
        batch_size=32,
        num_workers=4,
    )
    test_loader = loaders["test"]
    val_loader = loaders["val"]

    # Collect test predictions
    all_style_logits = []
    all_style_labels = []
    all_concept_logits = []
    all_concept_labels = []

    print("🔄 Running test inference...")
    with torch.no_grad():
        for images, style_labels, concept_labels in test_loader:
            images = images.to(device)
            concept_logits, style_logits = model(images)

            all_style_logits.append(style_logits.cpu())
            all_style_labels.append(style_labels)
            all_concept_logits.append(concept_logits.cpu())
            all_concept_labels.append(concept_labels)

    # Concatenate
    all_style_logits = torch.cat(all_style_logits, dim=0)
    all_style_labels = torch.cat(all_style_labels, dim=0)
    all_concept_logits = torch.cat(all_concept_logits, dim=0)
    all_concept_labels = torch.cat(all_concept_labels, dim=0)

    # Compute probabilities
    style_probs = torch.sigmoid(all_style_logits)
    concept_probs = torch.sigmoid(all_concept_logits)

    # Find optimal thresholds
    print("🎯 Finding optimal thresholds...")
    optimal_thresholds = find_optimal_thresholds(model, val_loader, device)

    # Save everything
    print("💾 Saving predictions...")
    os.makedirs("temp_evaluation", exist_ok=True)

    # Save as numpy arrays
    np.save("temp_evaluation/style_probs.npy", style_probs.numpy())
    np.save("temp_evaluation/style_labels.npy", all_style_labels.numpy())
    np.save("temp_evaluation/concept_probs.npy", concept_probs.numpy())
    np.save("temp_evaluation/concept_labels.npy", all_concept_labels.numpy())
    np.save("temp_evaluation/optimal_thresholds.npy", optimal_thresholds)

    # Save shapes and info
    info = {
        "style_probs_shape": list(style_probs.shape),
        "style_labels_shape": list(all_style_labels.shape),
        "concept_probs_shape": list(concept_probs.shape),
        "concept_labels_shape": list(all_concept_labels.shape),
        "optimal_thresholds_shape": list(optimal_thresholds.shape),
        "model_path": "model/cbm/models/cbm_decorrelated_best.pth",
        "n_concepts": 36,
        "n_classes": 18,
    }

    with open("temp_evaluation/info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("✅ Predictions saved to temp_evaluation/")
    print(f"   Style: {style_probs.shape} probs, {all_style_labels.shape} labels")
    print(
        f"   Concepts: {concept_probs.shape} probs, {all_concept_labels.shape} labels"
    )
    print(f"   Thresholds: {optimal_thresholds.shape}")


if __name__ == "__main__":
    collect_predictions()
