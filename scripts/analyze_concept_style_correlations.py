#!/usr/bin/env python3
"""
Quick analysis of concept-style correlations on a sample of images.
Start small for iteration, then scale up.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from tqdm import tqdm
from model.cbm_model import create_cbm_model
from model.cbm.concept_dataset import get_concept_data_loaders


def load_cbm_model():
    """Load the trained CBM model"""
    print("Loading CBM model...")

    # Create model
    model = create_cbm_model()

    # Load checkpoint
    checkpoint_path = "model/cbm/models/cbm_weighted_best.pth"
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Model not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ Model loaded from {checkpoint_path}")
    return model


def get_class_names():
    """Get art style class names"""
    with open("model/class_names.txt", "r") as f:
        return [line.strip() for line in f.readlines()]


def get_concept_names():
    """Get concept names"""
    with open("model/cbm/data/final_concepts.json", "r") as f:
        return json.load(f)["selected_concepts"]


def collect_predictions(model, dataloader, max_samples=200):
    """Collect concept and style predictions on a sample"""
    print(f"Collecting predictions (max {max_samples} samples)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_concept_preds = []
    all_style_preds = []
    all_concept_labels = []
    all_style_labels = []

    samples_processed = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            if samples_processed >= max_samples:
                break

            images, style_labels, concept_labels = batch
            images = images.to(device)

            # Forward pass
            concept_logits, style_logits = model(images)

            # Convert to probabilities
            concept_probs = torch.sigmoid(concept_logits)
            style_probs = torch.sigmoid(style_logits)

            # Store predictions and labels
            all_concept_preds.append(concept_probs.cpu().numpy())
            all_style_preds.append(style_probs.cpu().numpy())
            all_concept_labels.append(concept_labels.numpy())
            all_style_labels.append(style_labels.numpy())

            samples_processed += images.shape[0]

    # Concatenate all batches
    concept_preds = np.vstack(all_concept_preds)
    style_preds = np.vstack(all_style_preds)
    concept_labels = np.vstack(all_concept_labels)
    style_labels = np.vstack(all_style_labels)

    print(f"✅ Collected {concept_preds.shape[0]} samples")
    print(f"   Concept predictions: {concept_preds.shape}")
    print(f"   Style predictions: {style_preds.shape}")

    return concept_preds, style_preds, concept_labels, style_labels


def analyze_correlations(concept_preds, style_preds, concept_names, class_names):
    """Analyze concept-style correlations"""
    print("\n📊 Computing concept-style correlations...")

    # Compute correlation matrix between all concepts and styles
    # Shape: (n_concepts, n_styles)
    correlations = np.corrcoef(concept_preds.T, style_preds.T)
    n_concepts = concept_preds.shape[1]

    # Extract concept-style block
    concept_style_corr = correlations[:n_concepts, n_concepts:]

    print(f"Correlation matrix shape: {concept_style_corr.shape}")

    # Find top correlations for each style
    print("\n🎯 Top concept correlations by art style:")
    results = {}

    for style_idx, style_name in enumerate(class_names):
        # Get top 5 concepts for this style
        top_concept_indices = np.argsort(np.abs(concept_style_corr[:, style_idx]))[-5:][
            ::-1
        ]

        print(f"\n{style_name}:")
        style_results = []

        for concept_idx in top_concept_indices:
            corr = concept_style_corr[concept_idx, style_idx]
            concept_name = concept_names[concept_idx]

            if abs(corr) > 0.1:  # Only show meaningful correlations
                print(f"  {corr:+.3f} - {concept_name}")
                style_results.append({"concept": concept_name, "correlation": corr})

        results[style_name] = style_results

    return concept_style_corr, results


def create_heatmap(
    concept_style_corr,
    concept_names,
    class_names,
    save_path="concept_style_heatmap.png",
):
    """Create correlation heatmap"""
    print(f"\n📈 Creating correlation heatmap...")

    plt.figure(figsize=(14, 10))

    # Create heatmap
    sns.heatmap(
        concept_style_corr,
        xticklabels=class_names,
        yticklabels=concept_names,
        cmap="RdBu_r",
        center=0,
        annot=False,
        fmt=".2f",
        cbar_kws={"label": "Correlation"},
    )

    plt.title("Concept-Style Correlations", size=16, pad=20)
    plt.xlabel("Art Styles", size=12)
    plt.ylabel("Visual Concepts", size=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Heatmap saved to {save_path}")

    plt.show()


def save_results(concept_style_corr, concept_names, class_names, results):
    """Save detailed results"""

    # Save correlation matrix
    corr_df = pd.DataFrame(concept_style_corr, index=concept_names, columns=class_names)
    corr_df.to_csv("concept_style_correlations.csv")

    # Save top correlations summary
    with open("concept_style_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Results saved:")
    print("   - concept_style_correlations.csv (full matrix)")
    print("   - concept_style_summary.json (top correlations)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze concept-style correlations")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum samples to process (default: 200)",
    )
    parser.add_argument(
        "--dataset",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to use (default: val)",
    )

    args = parser.parse_args()

    print("🔍 CONCEPT-STYLE CORRELATION ANALYSIS")
    print("=" * 50)

    # Load model
    model = load_cbm_model()

    # Load data
    print(f"Loading {args.dataset} dataset...")
    loaders = get_concept_data_loaders(batch_size=16)
    dataloader = loaders[args.dataset]

    # Get names
    concept_names = get_concept_names()
    class_names = get_class_names()

    print(f"📋 Dataset info:")
    print(f"   Concepts: {len(concept_names)}")
    print(f"   Art styles: {len(class_names)}")

    # Collect predictions
    concept_preds, style_preds, concept_labels, style_labels = collect_predictions(
        model, dataloader, max_samples=args.max_samples
    )

    # Analyze correlations
    concept_style_corr, results = analyze_correlations(
        concept_preds, style_preds, concept_names, class_names
    )

    # Create visualization
    create_heatmap(concept_style_corr, concept_names, class_names)

    # Save results
    save_results(concept_style_corr, concept_names, class_names, results)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
