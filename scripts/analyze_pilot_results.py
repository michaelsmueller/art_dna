"""
Analyze existing pilot concept extraction results.
Skips kappa calculation (sklearn issue) but analyzes frequencies and correlations.
"""

import json
import numpy as np
from collections import defaultdict


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


def analyze_score_distribution(all_results, concepts):
    """Analyze the distribution of scores across the 4-level scale."""
    score_counts = {0: 0, 0.3: 0, 0.7: 0, 1.0: 0}
    total_scores = 0

    for results in all_results.values():
        for img_path, scores in results.items():
            for concept in concepts:
                score = scores.get(concept, 0)
                # Round to nearest expected value
                rounded_score = min([0, 0.3, 0.7, 1.0], key=lambda x: abs(x - score))
                score_counts[rounded_score] += 1
                total_scores += 1

    return score_counts, total_scores


def main():
    # Load pilot results
    try:
        with open("model/cbm/pilot_results.json", "r") as f:
            pilot_data = json.load(f)
    except FileNotFoundError:
        print("❌ No pilot results found. Run pilot_concept_extraction.py first.")
        return

    images = pilot_data["images"]
    results_1 = pilot_data["run1"]
    results_2 = pilot_data["run2"]
    concepts = pilot_data["concepts"]

    print("📊 PILOT RESULTS ANALYSIS")
    print("=" * 60)
    print(f"Images: {len(images)}")
    print(f"Concepts: {len(concepts)}")
    print(f"Runs completed: 2")

    # Load discriminative data for anti-Expressionism markers
    with open("model/cbm/discriminative_concepts.json", "r") as f:
        discriminative_data = json.load(f)
        anti_expressionism_markers = discriminative_data["anti_expressionism_markers"]

    print(f"Anti-Expressionism markers: {len(anti_expressionism_markers)}")
    print("=" * 60)

    # 1. Score distribution analysis
    all_results = {"run1": results_1, "run2": results_2}
    score_counts, total_scores = analyze_score_distribution(all_results, concepts)

    print(f"\n📈 Score Distribution:")
    for score, count in score_counts.items():
        percentage = (count / total_scores) * 100
        print(f"  {score}: {count:4d} ({percentage:5.1f}%)")

    # 2. Analyze frequencies
    presence_rates = analyze_concept_frequencies(all_results, concepts)

    # Find rare/common concepts
    rare = [(c, r) for c, r in presence_rates.items() if r < 0.02]
    common = [(c, r) for c, r in presence_rates.items() if r > 0.95]

    print(f"\n📊 Concept Frequencies:")
    print(f"  Rare (<2%): {len(rare)} concepts")
    print(f"  Common (>95%): {len(common)} concepts")

    if rare:
        print("\n  Rarest concepts:")
        for concept, rate in sorted(rare, key=lambda x: x[1])[:10]:
            print(f"    {concept}: {rate:.1%}")

    if common:
        print("\n  Most common concepts:")
        for concept, rate in sorted(common, key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {concept}: {rate:.1%}")

    # 3. Find high correlations
    high_corr = calculate_correlations(all_results, concepts)

    print(f"\n🔗 High Correlations (>0.9):")
    print(f"  Found {len(high_corr)} highly correlated pairs")
    if high_corr:
        print("\n  Top correlated pairs:")
        for c1, c2, corr in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[
            :10
        ]:
            print(f"    {c1} ↔ {c2}: {corr:.3f}")

    # 4. Anti-Expressionism analysis
    print(f"\n🎭 Anti-Expressionism Marker Analysis:")
    anti_expr_rates = {
        c: presence_rates[c] for c in anti_expressionism_markers if c in presence_rates
    }

    for marker, rate in sorted(
        anti_expr_rates.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {marker}: {rate:.1%}")

    # 5. Summary recommendations
    print("\n✅ ANALYSIS SUMMARY:")
    print(f"\n  📋 Concepts to consider removing:")
    if rare:
        print(f"    • {len(rare)} rare concepts (<2% presence)")
    if common:
        print(f"    • {len(common)} overly common concepts (>95% presence)")
    if high_corr:
        print(f"    • {len(high_corr)} highly correlated pairs (merge candidates)")

    print(f"\n  🎯 4-level scoring usage:")
    non_zero_three = score_counts[0.3] + score_counts[0.7] + score_counts[1.0]
    if non_zero_three > score_counts[0] * 0.1:  # If non-zero scores > 10% of total
        print(
            f"    ✓ 4-level scoring appears useful ({non_zero_three} non-zero scores)"
        )
    else:
        print(f"    ⚠ Consider binary scoring (few non-zero scores: {non_zero_three})")

    print(f"\n  💪 Anti-Expressionism balance:")
    avg_anti_expr_rate = (
        np.mean(list(anti_expr_rates.values())) if anti_expr_rates else 0
    )
    print(f"    Average presence: {avg_anti_expr_rate:.1%}")
    if avg_anti_expr_rate > 0.1:
        print(f"    ✓ Good balance - anti-markers are being detected")
    else:
        print(f"    ⚠ Low detection - may need stronger anti-markers")


if __name__ == "__main__":
    main()
