"""
Comprehensive verification of 8,027 concept extractions.
Ensures data quality before training.
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from pathlib import Path

# Load expected concepts (use the same ones that were actually extracted)
with open("model/cbm/discriminative_concepts.json", "r") as f:
    EXPECTED_CONCEPTS = json.load(f)["selected_concepts"]


def analyze_concept_distributions(results):
    """Analyze distribution of concept scores across all images."""
    concept_scores = defaultdict(list)

    for result in results:
        for concept, score in result["concepts"].items():
            concept_scores[concept].append(score)

    stats = {}
    for concept, scores in concept_scores.items():
        stats[concept] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "presence_rate": sum(1 for s in scores if s >= 0.7) / len(scores),
        }

    return stats


def analyze_concept_genre_correlations(results):
    """Analyze correlations between concepts and genres."""
    # Load genre data
    df = pd.read_csv("raw_data/final_df.csv")
    genre_columns = [
        col for col in df.columns if col not in ["image_path", "artist_name"]
    ]

    correlations = {}
    concept_genre_matrix = []

    for result in results:
        img_path = result["image_path"]

        # Find matching genre row
        genre_row = df[df["image_path"] == img_path]
        if len(genre_row) > 0:
            concept_vector = [result["concepts"].get(c, 0) for c in EXPECTED_CONCEPTS]
            genre_vector = genre_row[genre_columns].iloc[0].tolist()
            concept_genre_matrix.append(concept_vector + genre_vector)

    # Calculate correlations between concepts and genres
    matrix = np.array(concept_genre_matrix)
    concept_part = matrix[:, : len(EXPECTED_CONCEPTS)]
    genre_part = matrix[:, len(EXPECTED_CONCEPTS) :]

    for i, concept in enumerate(EXPECTED_CONCEPTS):
        correlations[concept] = {}
        for j, genre in enumerate(genre_columns):
            corr = np.corrcoef(concept_part[:, i], genre_part[:, j])[0, 1]
            if not np.isnan(corr):
                correlations[concept][genre] = corr

    return correlations


def spot_check_concept_quality(results, n_samples=20):
    """Manual spot check of concept quality on sample images."""
    import random

    # Select diverse samples
    samples = random.sample(results, min(n_samples, len(results)))

    quality_checks = []
    for sample in samples:
        # Get top 5 concepts for this image
        concepts = sample["concepts"]
        top_concepts = sorted(concepts.items(), key=lambda x: x[1], reverse=True)[:5]

        quality_checks.append(
            {
                "image": sample["image_path"],
                "top_concepts": top_concepts,
                "genres": sample.get("genres", {}),
            }
        )

    return quality_checks


def create_verification_report(concept_stats, genre_correlations, quality_samples):
    """Generate comprehensive verification report."""
    report = []

    report.append("\n📊 CONCEPT STATISTICS:")
    report.append("-" * 40)

    # Sort concepts by presence rate
    sorted_concepts = sorted(
        concept_stats.items(), key=lambda x: x[1]["presence_rate"], reverse=True
    )

    report.append(f"{'Concept':<25} {'Mean':<6} {'Std':<6} {'Present':<8}")
    report.append("-" * 50)

    for concept, stats in sorted_concepts:
        report.append(
            f"{concept:<25} {stats['mean']:<6.2f} {stats['std']:<6.2f} {stats['presence_rate']:<8.1%}"
        )

    # Find concerning concepts
    rare_concepts = [c for c, s in concept_stats.items() if s["presence_rate"] < 0.02]
    common_concepts = [c for c, s in concept_stats.items() if s["presence_rate"] > 0.95]

    if rare_concepts:
        report.append(f"\n⚠️  Rare concepts (<2%): {len(rare_concepts)}")
        for concept in rare_concepts[:5]:
            report.append(
                f"   {concept}: {concept_stats[concept]['presence_rate']:.1%}"
            )

    if common_concepts:
        report.append(f"\n⚠️  Overly common concepts (>95%): {len(common_concepts)}")
        for concept in common_concepts:
            report.append(
                f"   {concept}: {concept_stats[concept]['presence_rate']:.1%}"
            )

    # Top correlations
    report.append(f"\n🎯 TOP CONCEPT-GENRE CORRELATIONS:")
    report.append("-" * 40)

    all_correlations = []
    for concept, genre_corrs in genre_correlations.items():
        for genre, corr in genre_corrs.items():
            all_correlations.append((concept, genre, corr))

    top_correlations = sorted(all_correlations, key=lambda x: abs(x[2]), reverse=True)[
        :10
    ]

    for concept, genre, corr in top_correlations:
        report.append(f"{concept:<20} ↔ {genre:<15}: {corr:>6.3f}")

    # Quality samples
    report.append(f"\n🔍 QUALITY SPOT CHECKS:")
    report.append("-" * 40)

    for i, sample in enumerate(quality_samples[:5]):
        report.append(f"\nSample {i+1}: {sample['image'].split('/')[-1]}")
        report.append("Top concepts:")
        for concept, score in sample["top_concepts"]:
            report.append(f"   {concept}: {score:.2f}")

    # Print report
    for line in report:
        print(line)

    # Save to file
    with open("model/cbm/extraction_verification_report.txt", "w") as f:
        f.write("\n".join(report))

    print(f"\n📁 Full report saved to: model/cbm/extraction_verification_report.txt")


def verify_extraction_quality():
    """Main verification function."""
    print("🔍 CONCEPT EXTRACTION VERIFICATION")
    print("=" * 50)

    # Check if extraction file exists
    extraction_file = Path("model/cbm/full_extraction/full_concepts_complete.json")
    if not extraction_file.exists():
        print("❌ Extraction file not found!")
        print(f"   Expected: {extraction_file}")
        return False

    # Load extraction results
    with open(extraction_file, "r") as f:
        results = json.load(f)

    # 1. Basic integrity checks
    print(f"📊 Basic Checks:")
    print(f"   Total samples: {len(results)}")
    print(f"   Expected concepts: {len(EXPECTED_CONCEPTS)}")

    # 2. Check for missing/corrupted data
    missing_concepts = []
    corrupted_samples = []

    for result in results:
        actual_concepts = result.get("concepts", {})
        if len(actual_concepts) != len(EXPECTED_CONCEPTS):
            print(f"🔍 CORRUPTED SAMPLE DEBUG:")
            print(f"   Image: {result['image_path']}")
            print(f"   Expected concepts: {len(EXPECTED_CONCEPTS)}")
            print(f"   Actual concepts: {len(actual_concepts)}")
            print(
                f"   Missing concepts: {set(EXPECTED_CONCEPTS) - set(actual_concepts.keys())}"
            )
            print(
                f"   Extra concepts: {set(actual_concepts.keys()) - set(EXPECTED_CONCEPTS)}"
            )
            corrupted_samples.append(result["image_path"])

        for concept in EXPECTED_CONCEPTS:
            if concept not in result.get("concepts", {}):
                missing_concepts.append((result["image_path"], concept))

    if corrupted_samples:
        print(f"❌ Found {len(corrupted_samples)} corrupted samples")
        return False

    if missing_concepts:
        print(f"❌ Found {len(missing_concepts)} missing concepts")
        return False

    print("✅ Data integrity checks passed")

    # 3. Concept score distributions
    print("\n📈 Analyzing concept distributions...")
    concept_stats = analyze_concept_distributions(results)

    # 4. Genre correlation analysis
    print("🔗 Analyzing concept-genre correlations...")
    genre_correlations = analyze_concept_genre_correlations(results)

    # 5. Quality spot checks
    print("🔍 Running quality spot checks...")
    quality_samples = spot_check_concept_quality(results, n_samples=20)

    # 6. Generate verification report
    create_verification_report(concept_stats, genre_correlations, quality_samples)

    print(f"\n🎉 VERIFICATION COMPLETE!")
    print(f"✅ All {len(results)} samples verified successfully")
    print(f"✅ Ready for CBM training!")

    return True


if __name__ == "__main__":
    success = verify_extraction_quality()
    if not success:
        print("❌ Verification failed - check issues above")
        exit(1)
