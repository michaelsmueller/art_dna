"""
Pilot concept extraction with inter-rater reliability (kappa) testing.
Tests 20 diverse images to validate scoring consistency between AIs.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from collections import defaultdict

sys.path.append(".")

from model.cbm.concept_extraction import ConceptExtractor

# Load discriminative concepts
with open("model/cbm/discriminative_concepts.json", "r") as f:
    DISCRIMINATIVE_DATA = json.load(f)
    ALL_CONCEPTS = DISCRIMINATIVE_DATA["selected_concepts"]
    ANTI_EXPRESSIONISM_MARKERS = DISCRIMINATIVE_DATA["anti_expressionism_markers"]


def calculate_kappa_scores(results_1, results_2, concepts):
    """Calculate Cohen's kappa for each concept across two raters."""
    kappa_scores = {}

    for concept in concepts:
        scores_1 = []
        scores_2 = []

        for img_path in results_1:
            score_1 = results_1[img_path].get(concept, 0)
            score_2 = results_2[img_path].get(concept, 0)
            scores_1.append(score_1)
            scores_2.append(score_2)

        # Convert continuous scores to categorical labels for kappa
        # Treat 0, 0.3, 0.7, 1 as discrete categories
        labels = [0, 0.3, 0.7, 1.0]

        # Ensure scores are in the expected set
        scores_1_cat = [min(labels, key=lambda x: abs(x - s)) for s in scores_1]
        scores_2_cat = [min(labels, key=lambda x: abs(x - s)) for s in scores_2]

        # Calculate kappa
        if len(set(scores_1_cat)) > 1 or len(set(scores_2_cat)) > 1:
            # Use quadratic weights for ordered categories
            kappa = cohen_kappa_score(
                scores_1_cat, scores_2_cat, labels=labels, weights="quadratic"
            )
            kappa_scores[concept] = kappa
        else:
            # If no variance, kappa is undefined
            kappa_scores[concept] = None

    return kappa_scores


def analyze_concept_frequencies(all_results, concepts):
    """Calculate concept frequencies across all images."""
    frequencies = defaultdict(list)

    for results in all_results.values():
        for img_path, scores in results.items():
            for concept in concepts:
                score = scores.get(concept, 0)
                frequencies[concept].append(score)

    # Calculate presence rate (score > 0.3)
    presence_rates = {}
    for concept, scores in frequencies.items():
        presence_rate = sum(1 for s in scores if s >= 0.3) / len(scores)
        presence_rates[concept] = presence_rate

    return presence_rates


def calculate_correlations(all_results, concepts):
    """Calculate pairwise correlations between concepts."""
    # Build score matrix
    score_matrix = []

    for results in all_results.values():
        for img_path, scores in results.items():
            row = [scores.get(concept, 0) for concept in concepts]
            score_matrix.append(row)

    score_matrix = np.array(score_matrix)

    # Calculate correlations
    corr_matrix = np.corrcoef(score_matrix.T)

    # Find high correlations
    high_corr_pairs = []
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            corr = corr_matrix[i, j]
            if abs(corr) > 0.9:
                high_corr_pairs.append((concepts[i], concepts[j], corr))

    return high_corr_pairs


def select_pilot_images():
    """Select 20 diverse images for pilot testing."""
    df = pd.read_csv("raw_data/final_df.csv")
    genre_columns = [
        col for col in df.columns if col not in ["image_path", "artist_name"]
    ]

    selected_images = []

    # 1. Get 1 image from each genre (18 images)
    for genre in genre_columns:
        genre_df = df[df[genre] == 1]
        if len(genre_df) > 0:
            sample = genre_df.sample(n=1, random_state=42)
            selected_images.append(
                {
                    "path": sample["image_path"].iloc[0],
                    "genres": [genre],
                    "type": "single_genre",
                }
            )

    # 2. Add 2 multi-label images
    multi_label_df = df[df[genre_columns].sum(axis=1) > 1]
    if len(multi_label_df) >= 2:
        samples = multi_label_df.sample(n=2, random_state=42)
        for _, row in samples.iterrows():
            genres = [g for g in genre_columns if row[g] == 1]
            selected_images.append(
                {"path": row["image_path"], "genres": genres, "type": "multi_label"}
            )

    # Limit to exactly 20
    selected_images = selected_images[:20]

    print(f"🎯 Selected {len(selected_images)} pilot images:")
    print(
        f"  - Single genre: {sum(1 for img in selected_images if img['type'] == 'single_genre')}"
    )
    print(
        f"  - Multi-label: {sum(1 for img in selected_images if img['type'] == 'multi_label')}"
    )

    return selected_images


async def run_pilot_extraction(extractor, images, run_id="run1"):
    """Run concept extraction on pilot images."""
    results = {}

    print(f"\n🔄 Extracting concepts ({run_id})...")
    for i, img_info in enumerate(images):
        print(f"  Processing {i+1}/{len(images)}: {img_info['path']}")
        concepts = await extractor.extract_concepts_single(img_info["path"])
        results[img_info["path"]] = concepts

        # Brief pause to avoid rate limits
        if i < len(images) - 1:
            time.sleep(0.5)

    return results


def main():
    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("⚠️  Please set ANTHROPIC_API_KEY environment variable")
        exit(1)

    print("🧪 PILOT CONCEPT EXTRACTION TEST")
    print("=" * 60)
    print(f"Concepts: {len(ALL_CONCEPTS)} discriminative concepts")
    print(f"Scoring: 0 / 0.3 / 0.7 / 1")
    print(f"Anti-Expressionism markers: {len(ANTI_EXPRESSIONISM_MARKERS)}")
    print("=" * 60)

    # Select pilot images
    pilot_images = select_pilot_images()

    # Create extractor and override concepts
    extractor = ConceptExtractor(api_key)
    extractor.concepts = ALL_CONCEPTS  # Use only 40 discriminative concepts

    # Run extraction twice for kappa testing
    import asyncio

    # First run
    results_1 = asyncio.run(run_pilot_extraction(extractor, pilot_images, "run1"))

    # Second run (simulating different rater)
    print("\n⏳ Waiting 5 seconds before second run...")
    time.sleep(5)
    results_2 = asyncio.run(run_pilot_extraction(extractor, pilot_images, "run2"))

    # Save raw results
    pilot_data = {
        "images": pilot_images,
        "run1": results_1,
        "run2": results_2,
        "concepts": ALL_CONCEPTS,
    }

    with open("model/cbm/pilot_results.json", "w") as f:
        json.dump(pilot_data, f, indent=2)

    print("\n📊 PILOT ANALYSIS")
    print("=" * 60)

    # 1. Calculate kappa scores
    kappa_scores = calculate_kappa_scores(results_1, results_2, ALL_CONCEPTS)
    valid_kappas = [k for k in kappa_scores.values() if k is not None]
    avg_kappa = np.mean(valid_kappas) if valid_kappas else 0

    print(f"\n🎯 Inter-rater Agreement (Kappa):")
    print(f"  Average κ: {avg_kappa:.3f}")
    print(f"  κ >= 0.6: {sum(1 for k in valid_kappas if k >= 0.6)} concepts")
    print(f"  κ < 0.6: {sum(1 for k in valid_kappas if k < 0.6)} concepts")

    # Show worst agreement concepts
    worst_kappas = sorted(
        [(c, k) for c, k in kappa_scores.items() if k is not None], key=lambda x: x[1]
    )[:10]
    print("\n  Lowest agreement concepts:")
    for concept, kappa in worst_kappas:
        print(f"    {concept}: κ = {kappa:.3f}")

    # 2. Analyze frequencies
    all_results = {"run1": results_1, "run2": results_2}
    presence_rates = analyze_concept_frequencies(all_results, ALL_CONCEPTS)

    # Find rare/common concepts
    rare = [(c, r) for c, r in presence_rates.items() if r < 0.02]
    common = [(c, r) for c, r in presence_rates.items() if r > 0.95]

    print(f"\n📈 Concept Frequencies:")
    print(f"  Rare (<2%): {len(rare)} concepts")
    print(f"  Common (>95%): {len(common)} concepts")

    if rare:
        print("\n  Rarest concepts:")
        for concept, rate in sorted(rare, key=lambda x: x[1])[:5]:
            print(f"    {concept}: {rate:.1%}")

    # 3. Find high correlations
    high_corr = calculate_correlations(all_results, ALL_CONCEPTS)

    print(f"\n🔗 High Correlations (>0.9):")
    print(f"  Found {len(high_corr)} highly correlated pairs")
    if high_corr:
        for c1, c2, corr in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[
            :5
        ]:
            print(f"    {c1} ↔ {c2}: {corr:.3f}")

    # 4. Decision
    print("\n✅ PILOT DECISION:")
    if avg_kappa >= 0.6:
        print(f"  ✓ Average κ = {avg_kappa:.3f} >= 0.6 → Keep 4-level scoring")
    else:
        print(f"  ✗ Average κ = {avg_kappa:.3f} < 0.6 → Collapse to binary (0/1)")

    print(f"\n📁 Full results saved to: model/cbm/pilot_results.json")


if __name__ == "__main__":
    main()
