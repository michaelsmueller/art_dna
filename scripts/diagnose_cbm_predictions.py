#!/usr/bin/env python3
"""
Diagnose why F1 scores are 0 - analyze model predictions after 2 epochs.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from model.cbm_model import create_cbm_model
from model.cbm.concept_dataset import get_concept_data_loaders
import json


def diagnose_predictions():
    print("🔍 DIAGNOSING CBM PREDICTIONS")
    print("=" * 50)

    # Load model
    print("\n📊 Loading trained model and data...")
    model = create_cbm_model()
    model.eval()

    # Get data loader
    loaders = get_concept_data_loaders(batch_size=32, num_workers=0)

    # Analyze predictions on validation set
    all_style_probs = []
    all_concept_probs = []
    all_style_labels = []
    all_concept_labels = []

    print("\n🔍 Analyzing predictions on validation set...")
    with torch.no_grad():
        for i, (images, style_labels, concept_labels) in enumerate(loaders["val"]):
            if i >= 10:  # Analyze first 10 batches
                break

            # Get predictions
            concept_logits, style_logits = model(images)

            # Convert to probabilities
            style_probs = torch.sigmoid(style_logits)
            concept_probs = torch.sigmoid(concept_logits)

            all_style_probs.append(style_probs.cpu().numpy())
            all_concept_probs.append(concept_probs.cpu().numpy())
            all_style_labels.append(style_labels.cpu().numpy())
            all_concept_labels.append(concept_labels.cpu().numpy())

    # Concatenate all batches
    style_probs = np.vstack(all_style_probs)
    concept_probs = np.vstack(all_concept_probs)
    style_labels = np.vstack(all_style_labels)
    concept_labels = np.vstack(all_concept_labels)

    print(f"\n📊 Analyzed {len(style_probs)} samples")

    # Analyze style predictions
    print("\n🎨 STYLE PREDICTIONS ANALYSIS:")
    print(f"  Probability range: [{style_probs.min():.4f}, {style_probs.max():.4f}]")
    print(f"  Mean probability: {style_probs.mean():.4f}")
    print(f"  Predictions > 0.5: {(style_probs > 0.5).sum()} / {style_probs.size}")
    print(f"  Predictions > 0.3: {(style_probs > 0.3).sum()} / {style_probs.size}")
    print(f"  Predictions > 0.1: {(style_probs > 0.1).sum()} / {style_probs.size}")

    # Per-class analysis
    print("\n  Per-class maximum probabilities:")
    with open("model/class_names.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    for i, class_name in enumerate(class_names[:5]):  # Show first 5
        max_prob = style_probs[:, i].max()
        mean_prob = style_probs[:, i].mean()
        positive_rate = (style_labels[:, i] > 0).mean()
        print(
            f"    {class_name}: max={max_prob:.3f}, mean={mean_prob:.3f}, positive_rate={positive_rate:.3f}"
        )

    # Analyze concept predictions
    print("\n🧠 CONCEPT PREDICTIONS ANALYSIS:")
    print(
        f"  Probability range: [{concept_probs.min():.4f}, {concept_probs.max():.4f}]"
    )
    print(f"  Mean probability: {concept_probs.mean():.4f}")
    print(f"  Predictions > 0.5: {(concept_probs > 0.5).sum()} / {concept_probs.size}")
    print(f"  Predictions > 0.3: {(concept_probs > 0.3).sum()} / {concept_probs.size}")
    print(f"  Predictions > 0.1: {(concept_probs > 0.1).sum()} / {concept_probs.size}")

    # Suggest threshold
    print("\n💡 RECOMMENDATIONS:")

    # Find optimal threshold for a few examples
    best_threshold_style = 0.5
    best_threshold_concept = 0.5

    for threshold in np.arange(0.05, 0.5, 0.05):
        style_preds = (style_probs > threshold).astype(int)
        concept_preds = (concept_probs > threshold).astype(int)

        # Check if any predictions are made
        if style_preds.sum() > 0:
            best_threshold_style = min(best_threshold_style, threshold)
        if concept_preds.sum() > 0:
            best_threshold_concept = min(best_threshold_concept, threshold)

    print(f"  - Model outputs very low probabilities (normal for early training)")
    print(f"  - Suggested style threshold for testing: {best_threshold_style:.2f}")
    print(f"  - Suggested concept threshold for testing: {best_threshold_concept:.2f}")
    print(f"  - Continue training for 8-10 more epochs for probabilities to calibrate")

    # Check class balance
    print("\n📊 CLASS BALANCE CHECK:")
    style_positive_rate = (style_labels > 0).mean(axis=0)
    print(
        f"  Style positive rates: min={style_positive_rate.min():.3f}, max={style_positive_rate.max():.3f}, mean={style_positive_rate.mean():.3f}"
    )

    concept_positive_rate = (concept_labels > 0).mean(axis=0)
    print(
        f"  Concept positive rates: min={concept_positive_rate.min():.3f}, max={concept_positive_rate.max():.3f}, mean={concept_positive_rate.mean():.3f}"
    )

    if style_positive_rate.min() < 0.01:
        print("\n  ⚠️  Warning: Some style classes are very rare (<1% positive rate)")
        print("     Consider using weighted loss or focal loss")

    return style_probs, concept_probs


if __name__ == "__main__":
    diagnose_predictions()
