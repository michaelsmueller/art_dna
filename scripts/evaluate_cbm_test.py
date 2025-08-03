#!/usr/bin/env python3
"""
Evaluate the trained CBM model on the test set using optimal thresholds.
Reports detailed metrics and saves results.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import f1_score, classification_report, accuracy_score
from tqdm import tqdm

from model.cbm_model import ConceptBottleneckModel
from model.cbm.concept_dataset import get_concept_data_loaders


def load_model_and_thresholds(checkpoint_path="model/cbm/cbm_weighted_best.pth"):
    """Load the trained model and optimal thresholds."""
    print("🧠 Loading trained model...")

    model = ConceptBottleneckModel(
        n_concepts=37, n_classes=18, backbone_weights=None, freeze_backbone=False
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    optimal_thresholds = checkpoint.get("optimal_thresholds", None)
    if optimal_thresholds is None:
        print("⚠️  No optimal thresholds found, using default 0.5")
        optimal_thresholds = np.ones(18) * 0.5
    else:
        print(f"✅ Loaded optimal thresholds")

    return model, optimal_thresholds, checkpoint


def evaluate_on_test_set(model, test_loader, optimal_thresholds, device):
    """Evaluate model on test set with optimal thresholds."""
    print("\n📊 Evaluating on test set...")

    all_style_preds = []
    all_style_labels = []
    all_concept_preds = []
    all_concept_labels = []

    with torch.no_grad():
        for images, style_labels, concept_labels in tqdm(
            test_loader, desc="Evaluating"
        ):
            images = images.to(device)

            # Get predictions
            concept_logits, style_logits = model(images)

            # Apply sigmoid
            style_probs = torch.sigmoid(style_logits).cpu().numpy()
            concept_probs = torch.sigmoid(concept_logits).cpu().numpy()

            # Apply optimal thresholds for styles
            style_preds = (style_probs >= optimal_thresholds).astype(int)

            # Use 0.5 threshold for concepts
            concept_preds = (concept_probs >= 0.5).astype(int)

            all_style_preds.append(style_preds)
            all_style_labels.append(style_labels.numpy())
            all_concept_preds.append(concept_preds)
            all_concept_labels.append(concept_labels.numpy())

    # Concatenate all predictions
    style_preds = np.vstack(all_style_preds)
    style_labels = np.vstack(all_style_labels)
    concept_preds = np.vstack(all_concept_preds)
    concept_labels = np.vstack(all_concept_labels)

    return style_preds, style_labels, concept_preds, concept_labels


def compute_detailed_metrics(preds, labels, class_names, metric_type="Style"):
    """Compute detailed metrics for predictions."""
    print(f"\n📈 {metric_type} Metrics:")

    # Overall metrics
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)

    print(f"  F1 Macro:    {f1_macro:.3f}")
    print(f"  F1 Weighted: {f1_weighted:.3f}")
    print(f"  F1 Micro:    {f1_micro:.3f}")

    # Per-class metrics
    print(f"\n📊 Per-class {metric_type} Performance:")
    print(f"{'Class':<20} {'Support':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 60)

    for i, class_name in enumerate(class_names):
        support = labels[:, i].sum()
        f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)

        # Calculate precision and recall manually
        tp = ((preds[:, i] == 1) & (labels[:, i] == 1)).sum()
        fp = ((preds[:, i] == 1) & (labels[:, i] == 0)).sum()
        fn = ((preds[:, i] == 0) & (labels[:, i] == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(
            f"{class_name:<20} {support:>8} {f1:>8.3f} {precision:>10.3f} {recall:>8.3f}"
        )

    return {"f1_macro": f1_macro, "f1_weighted": f1_weighted, "f1_micro": f1_micro}


def analyze_threshold_impact(
    style_probs, style_labels, optimal_thresholds, class_names
):
    """Analyze the impact of optimal thresholds vs default 0.5."""
    print("\n🔍 Threshold Impact Analysis:")

    # Compare with default 0.5 threshold
    default_preds = (style_probs >= 0.5).astype(int)
    optimal_preds = (style_probs >= optimal_thresholds).astype(int)

    default_f1 = f1_score(
        style_labels, default_preds, average="weighted", zero_division=0
    )
    optimal_f1 = f1_score(
        style_labels, optimal_preds, average="weighted", zero_division=0
    )

    print(f"  F1 with 0.5 threshold:     {default_f1:.3f}")
    print(f"  F1 with optimal thresholds: {optimal_f1:.3f}")
    print(
        f"  Improvement:                {optimal_f1 - default_f1:.3f} ({(optimal_f1/default_f1 - 1)*100:.1f}%)"
    )

    # Show threshold values
    print("\n📊 Optimal Thresholds by Class:")
    for i, (class_name, threshold) in enumerate(zip(class_names, optimal_thresholds)):
        if i < 10:  # Show first 10
            print(f"  {class_name:<20}: {threshold:.2f}")
    if len(class_names) > 10:
        print(f"  ... and {len(class_names) - 10} more")


def get_class_names():
    """Get style class names."""
    return [
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


def get_concept_names():
    """Load concept names."""
    with open("model/cbm/data/final_concepts.json", "r") as f:
        return json.load(f)["selected_concepts"]


def main():
    print("🧪 CBM TEST SET EVALUATION")
    print("=" * 50)

    # Check if model exists
    checkpoint_path = "model/cbm/cbm_weighted_best.pth"
    if not Path(checkpoint_path).exists():
        print(f"❌ No checkpoint found at {checkpoint_path}")
        print("   Please complete training first!")
        return

    # Load model and data
    model, optimal_thresholds, checkpoint = load_model_and_thresholds(checkpoint_path)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    # Load test data
    print("\n📊 Loading test data...")
    loaders = get_concept_data_loaders(batch_size=32, num_workers=0)
    test_loader = loaders["test"]
    print(f"  Test samples: {len(test_loader.dataset)}")

    # Get predictions
    style_preds, style_labels, concept_preds, concept_labels = evaluate_on_test_set(
        model, test_loader, optimal_thresholds, device
    )

    # Also get raw probabilities for threshold analysis
    all_style_probs = []
    with torch.no_grad():
        for images, _, _ in test_loader:
            images = images.to(device)
            _, style_logits = model(images)
            style_probs = torch.sigmoid(style_logits).cpu().numpy()
            all_style_probs.append(style_probs)
    style_probs = np.vstack(all_style_probs)

    # Compute metrics
    class_names = get_class_names()
    concept_names = get_concept_names()

    style_metrics = compute_detailed_metrics(
        style_preds, style_labels, class_names, "Style"
    )
    # Debug concept data types
    print(f"Concept preds shape: {concept_preds.shape}, dtype: {concept_preds.dtype}")
    print(
        f"Concept labels shape: {concept_labels.shape}, dtype: {concept_labels.dtype}"
    )
    print(f"Concept preds range: [{concept_preds.min()}, {concept_preds.max()}]")
    print(f"Concept labels range: [{concept_labels.min()}, {concept_labels.max()}]")

    # Ensure concept labels are also binary integers
    concept_labels_binary = (concept_labels > 0.5).astype(int)

    # Only analyze first 10 concepts to keep output manageable
    concept_metrics = compute_detailed_metrics(
        concept_preds[:, :10],
        concept_labels_binary[:, :10],
        concept_names[:10],
        "Concept (Top 10)",
    )

    # Analyze threshold impact
    analyze_threshold_impact(style_probs, style_labels, optimal_thresholds, class_names)

    # Training summary
    print("\n📋 Training Summary:")
    if "history" in checkpoint:
        history = checkpoint["history"]
        final_epoch = len(history["train"])
        print(f"  Total epochs trained: {final_epoch}")
        print(f"  Final train loss: {history['train'][-1]['losses']['total']:.3f}")
        print(f"  Final val loss: {history['val'][-1]['losses']['total']:.3f}")

    # Save results
    results = {
        "test_metrics": {"style": style_metrics, "concept": concept_metrics},
        "optimal_thresholds": optimal_thresholds.tolist(),
        "model_info": {
            "checkpoint": checkpoint_path,
            "n_test_samples": len(test_loader.dataset),
        },
    }

    results_path = "model/cbm/test_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {results_path}")

    # Final decision
    print("\n🎯 FINAL EVALUATION:")
    print(f"  Test F1 (weighted): {style_metrics['f1_weighted']:.3f}")
    print(f"  Target (VGG16):     0.431")

    if style_metrics["f1_weighted"] >= 0.55:
        print("\n🎉 SUCCESS! Model exceeds baseline by >27%")
        print("   Ready for Vertex AI deployment!")
    elif style_metrics["f1_weighted"] >= 0.45:
        print("\n✅ Good performance, close to baseline")
        print("   Consider focal loss or lower weight cap for improvement")
    else:
        print("\n⚠️  Below expectations")
        print("   Recommend switching to staged training approach")


if __name__ == "__main__":
    main()
