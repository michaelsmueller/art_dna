#!/usr/bin/env python3
"""
Compare our weight calculation with sklearn's balanced approach.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from model.pytorch_data_loader import calculate_pos_weights
import torch


def main():
    print("🔍 Comparing Weight Calculation Approaches")
    print("=" * 50)

    # Load data
    df = pd.read_csv("raw_data/final_df.csv")
    style_columns = [
        col for col in df.columns if col not in ["image_path", "artist_name"]
    ]

    print(f"\n📊 Dataset: {len(df)} images, {len(style_columns)} styles")

    # Our current approach (from calculate_pos_weights)
    our_weights = calculate_pos_weights("raw_data/final_df.csv")

    # Sklearn balanced approach (like Kristina used)
    sklearn_weights = []
    for i, col in enumerate(style_columns):
        labels = df[col].values
        # Compute balanced weights
        class_weight = compute_class_weight(
            class_weight="balanced", classes=np.array([0, 1]), y=labels
        )
        # Get positive class weight
        sklearn_weights.append(class_weight[1])

    sklearn_weights = torch.tensor(sklearn_weights)

    # Compare
    print("\n📊 Weight Comparison:")
    print(f"{'Style':<20} {'Our Weight':>12} {'SKlearn':>12} {'Ratio':>8}")
    print("-" * 55)

    for i, col in enumerate(style_columns):
        our_w = our_weights[i].item()
        sk_w = sklearn_weights[i].item()
        ratio = our_w / sk_w if sk_w > 0 else 0
        print(f"{col:<20} {our_w:>12.2f} {sk_w:>12.2f} {ratio:>8.2f}")

    print(f"\n📈 Summary Statistics:")
    print(
        f"Our approach:     min={our_weights.min():.2f}, max={our_weights.max():.2f}, mean={our_weights.mean():.2f}"
    )
    print(
        f"SKlearn balanced: min={sklearn_weights.min():.2f}, max={sklearn_weights.max():.2f}, mean={sklearn_weights.mean():.2f}"
    )

    # Calculate what happens with capping
    our_capped = torch.clamp(our_weights, max=5.0)
    sklearn_capped = torch.clamp(sklearn_weights, max=5.0)

    print(f"\nWith cap=5.0:")
    print(
        f"Our capped:       min={our_capped.min():.2f}, max={our_capped.max():.2f}, mean={our_capped.mean():.2f}"
    )
    print(
        f"SKlearn capped:   min={sklearn_capped.min():.2f}, max={sklearn_capped.max():.2f}, mean={sklearn_capped.mean():.2f}"
    )

    # Show which classes get capped
    capped_classes = [
        col for i, col in enumerate(style_columns) if our_weights[i] > 5.0
    ]
    print(f"\n⚠️  Classes that get capped in our approach: {capped_classes}")


if __name__ == "__main__":
    main()
