#!/usr/bin/env python3
"""
Load saved CBM predictions and compute metrics with debugging.
"""

import sys
import numpy as np
import json
from sklearn.metrics import f1_score

sys.path.append(".")


def debug_and_compute_metrics():
    """Load predictions and compute metrics with detailed debugging."""

    print("📊 Loading saved predictions...")

    # Load all data
    style_probs = np.load("temp_evaluation/style_probs.npy")
    style_labels = np.load("temp_evaluation/style_labels.npy")
    concept_probs = np.load("temp_evaluation/concept_probs.npy")
    concept_labels = np.load("temp_evaluation/concept_labels.npy")
    optimal_thresholds = np.load("temp_evaluation/optimal_thresholds.npy")

    with open("temp_evaluation/info.json", "r") as f:
        info = json.load(f)

    print(f"✅ Loaded data:")
    print(f"   Style: {style_probs.shape} probs, {style_labels.shape} labels")
    print(f"   Concepts: {concept_probs.shape} probs, {concept_labels.shape} labels")
    print(f"   Thresholds: {optimal_thresholds.shape}")

    # Debug data types and values
    print(f"\n🔍 Data inspection:")
    print(
        f"Style labels dtype: {style_labels.dtype}, unique: {np.unique(style_labels)}"
    )
    print(
        f"Style probs dtype: {style_probs.dtype}, range: [{style_probs.min():.3f}, {style_probs.max():.3f}]"
    )
    print(
        f"Concept labels dtype: {concept_labels.dtype}, unique: {np.unique(concept_labels)}"
    )
    print(
        f"Concept probs dtype: {concept_probs.dtype}, range: [{concept_probs.min():.3f}, {concept_probs.max():.3f}]"
    )

    # STEP 1: Style metrics (this worked in debug)
    print(f"\n📈 Computing Style Metrics...")

    # Binarize style labels
    style_labels_binary = (style_labels > 0.5).astype(int)
    print(f"Style labels after binarization: unique={np.unique(style_labels_binary)}")

    # Apply optimal thresholds
    style_preds = (style_probs > optimal_thresholds).astype(int)
    print(
        f"Style predictions: shape={style_preds.shape}, unique={np.unique(style_preds)}"
    )

    # Test style F1
    try:
        style_f1 = f1_score(style_labels_binary, style_preds, average="weighted")
        print(f"✅ Style F1 (weighted): {style_f1:.3f}")
    except Exception as e:
        print(f"❌ Style F1 failed: {e}")
        return  # Stop here if style fails

    # STEP 2: Concept metrics (this is where it fails)
    print(f"\n📈 Computing Concept Metrics...")

    print(f"BEFORE adjustment: concept_labels.shape = {concept_labels.shape}")
    print(f"BEFORE adjustment: concept_probs.shape = {concept_probs.shape}")

    # Need to adjust concept labels from 37 -> 36
    # Load concept info to get keep_indices
    with open("model/cbm/data/final_concepts.json", "r") as f:
        concept_info = json.load(f)

    selected_concepts = concept_info["selected_concepts"]
    keep_indices = [
        i
        for i, concept in enumerate(selected_concepts)
        if concept != "angular_fragmentation"
    ]

    print(f"keep_indices: {len(keep_indices)} indices: {keep_indices[:5]}...")

    # Adjust concept labels
    concept_labels_adjusted = concept_labels[:, keep_indices]
    print(
        f"AFTER adjustment: concept_labels_adjusted.shape = {concept_labels_adjusted.shape}"
    )

    # Binarize
    concept_labels_binary = (concept_labels_adjusted > 0.5).astype(int)
    concept_preds = (concept_probs > 0.5).astype(int)

    print(
        f"concept_labels_binary: shape={concept_labels_binary.shape}, unique={np.unique(concept_labels_binary)}"
    )
    print(
        f"concept_preds: shape={concept_preds.shape}, unique={np.unique(concept_preds)}"
    )

    # Test shapes match
    if concept_labels_binary.shape != concept_preds.shape:
        print(f"❌ Shape mismatch!")
        print(f"   Labels: {concept_labels_binary.shape}")
        print(f"   Preds:  {concept_preds.shape}")
        return

    # Test concept F1
    try:
        concept_f1 = f1_score(concept_labels_binary, concept_preds, average="weighted")
        print(f"✅ Concept F1 (weighted): {concept_f1:.3f}")
    except Exception as e:
        print(f"❌ Concept F1 failed: {e}")
        # Additional debugging
        print(f"Labels dtype: {concept_labels_binary.dtype}")
        print(f"Preds dtype: {concept_preds.dtype}")
        print(f"Labels shape: {concept_labels_binary.shape}")
        print(f"Preds shape: {concept_preds.shape}")

        # Try with just first 10 samples
        try:
            test_f1 = f1_score(
                concept_labels_binary[:10], concept_preds[:10], average="weighted"
            )
            print(f"Small sample F1 works: {test_f1:.3f}")
        except Exception as e2:
            print(f"Even small sample fails: {e2}")

        return

    # If we get here, both work!
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Style F1 (weighted): {style_f1:.3f}")
    print(f"Concept F1 (weighted): {concept_f1:.3f}")

    # Compare with v1.0
    try:
        with open("model/cbm/test_evaluation_results.json", "r") as f:
            v1_results = json.load(f)
        v1_f1 = v1_results.get("f1_weighted", 0.601)
        print(f"\n📊 Comparison with v1.0:")
        print(f"v1.0 Style F1: {v1_f1:.3f}")
        print(f"v1.1 Style F1: {style_f1:.3f}")
        print(f"Change: {style_f1 - v1_f1:+.3f} ({((style_f1/v1_f1 - 1)*100):+.1f}%)")
    except:
        print(f"\n⚠️  Could not load v1.0 results for comparison")


if __name__ == "__main__":
    debug_and_compute_metrics()
