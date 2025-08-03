#!/usr/bin/env python3
"""
Pre-calculate and cache concept weights to speed up training iterations.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import json
from model.cbm.concept_dataset import get_concept_data_loaders
from tqdm import tqdm


def calculate_and_cache_concept_weights():
    print("📊 Calculating concept weights from full training set...")

    # Load data
    loaders = get_concept_data_loaders(batch_size=32, num_workers=0)
    n_concepts = 37
    weight_cap = 5.0

    concept_pos_counts = torch.zeros(n_concepts)
    total_samples = 0

    with torch.no_grad():
        for images, _, concept_labels in tqdm(
            loaders["train"], desc="Scanning concepts"
        ):
            concept_pos_counts += (concept_labels > 0.5).float().sum(dim=0)
            total_samples += len(concept_labels)

    # Calculate rates and weights
    concept_pos_rates = concept_pos_counts / total_samples
    concept_neg_rates = 1 - concept_pos_rates
    concept_pos_weights = concept_neg_rates / (concept_pos_rates + 1e-6)
    concept_pos_weights = torch.clamp(concept_pos_weights, min=0.1, max=weight_cap)

    # Save results
    cache_data = {
        "concept_pos_rates": concept_pos_rates.tolist(),
        "concept_pos_weights": concept_pos_weights.tolist(),
        "total_samples": total_samples,
        "weight_cap": weight_cap,
    }

    cache_path = "model/cbm/concept_weights_cache.json"
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)

    print(f"✅ Cached to {cache_path}")
    print(f"   Scanned {total_samples} samples")
    print(
        f"   Concept positive rates: min={concept_pos_rates.min():.3f}, max={concept_pos_rates.max():.3f}"
    )
    print(
        f"   Concept weights: min={concept_pos_weights.min():.2f}, max={concept_pos_weights.max():.2f}"
    )

    return concept_pos_weights


if __name__ == "__main__":
    calculate_and_cache_concept_weights()
