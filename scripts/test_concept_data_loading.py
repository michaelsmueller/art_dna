#!/usr/bin/env python3
"""
Test concept data loading - verify images, styles, and concepts align correctly.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.cbm.concept_dataset import get_concept_data_loaders
import json

print("🧪 TESTING CONCEPT DATA INTEGRATION")
print("=" * 50)

# Load concept names for reference
with open("model/cbm/data/final_concepts.json", "r") as f:
    concept_info = json.load(f)
concept_names = concept_info["selected_concepts"]

# Load genre names
with open("model/class_names.txt", "r") as f:
    genre_names = [line.strip() for line in f.readlines()]

# Create data loaders with small batch
print("\n📊 Creating concept-aware data loaders...")
loaders = get_concept_data_loaders(batch_size=4, num_workers=0)

# Test train loader
train_loader = loaders["train"]
print(f"\n✅ Train loader created with {len(train_loader)} batches")

# Get first batch
print("\n🔍 Inspecting first batch...")
images, style_labels, concept_labels = next(iter(train_loader))

print(f"\n📐 Tensor Shapes:")
print(f"  Images: {images.shape} (batch, channels, height, width)")
print(f"  Style labels: {style_labels.shape} (batch, n_genres)")
print(f"  Concept labels: {concept_labels.shape} (batch, n_concepts)")

# Analyze each sample in batch
print("\n📋 Sample Analysis:")
for i in range(min(4, len(images))):
    print(f"\n  Sample {i+1}:")

    # Style analysis
    active_styles = [genre_names[j] for j, val in enumerate(style_labels[i]) if val > 0]
    print(f"    Genres: {', '.join(active_styles) if active_styles else 'None'}")

    # Concept analysis
    concept_scores = concept_labels[i]
    active_concepts = [
        (concept_names[j], concept_scores[j].item())
        for j in range(len(concept_names))
        if concept_scores[j] > 0.5
    ]

    print(f"    Active concepts ({len(active_concepts)}):")
    for concept, score in sorted(active_concepts, key=lambda x: x[1], reverse=True)[:5]:
        print(f"      - {concept}: {score:.2f}")

    # Concept statistics
    print(f"    Concept stats:")
    print(f"      - Total active: {(concept_scores > 0.5).sum().item()}")
    print(f"      - Mean score: {concept_scores.mean().item():.3f}")
    print(f"      - Max score: {concept_scores.max().item():.3f}")

# Overall batch statistics
print("\n📊 Batch Statistics:")
print(
    f"  Multi-label styles: {(style_labels.sum(dim=1) > 1).sum().item()}/{len(style_labels)} images"
)
print(
    f"  Avg concepts per image: {(concept_labels > 0.5).sum(dim=1).float().mean().item():.1f}"
)
print(
    f"  Concept score range: [{concept_labels.min().item():.2f}, {concept_labels.max().item():.2f}]"
)

# Test validation loader briefly
print("\n🔍 Testing validation loader...")
val_loader = loaders["val"]
val_images, val_styles, val_concepts = next(iter(val_loader))
print(f"✅ Val loader working: {val_images.shape[0]} samples loaded")

# Memory estimate
import torch

print("\n💾 Memory Usage Estimate:")
print(f"  Images: {images.element_size() * images.nelement() / 1024**2:.1f} MB")
print(
    f"  Concepts: {concept_labels.element_size() * concept_labels.nelement() / 1024:.1f} KB"
)
print(
    f"  Styles: {style_labels.element_size() * style_labels.nelement() / 1024:.1f} KB"
)

print("\n✅ CONCEPT DATA INTEGRATION TEST COMPLETE!")
print("\n🚀 Ready for next step: Basic CBM training infrastructure")
