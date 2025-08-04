# scripts/analyze_concept_discrimination.py
"""
Analyze which concepts are most discriminative for different art genres.
Identifies concepts that best distinguish between styles while avoiding
overclassification of Expressionism.
"""

import sys
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_selection import chi2, mutual_info_classif

sys.path.append(".")

from model.cbm.concept_extraction import ALL_CONCEPTS, CONCEPT_CATEGORIES


def load_genre_data():
    """Load dataset and get genre information."""
    df = pd.read_csv("raw_data/final_df.csv")
    genre_columns = [
        col for col in df.columns if col not in ["image_path", "artist_name"]
    ]

    # Get genre distribution
    genre_counts = {}
    for genre in genre_columns:
        genre_counts[genre] = df[genre].sum()

    print("📊 Genre Distribution:")
    for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {genre}: {count} images")

    return df, genre_columns, genre_counts


def simulate_concept_presence(df, genre_columns):
    """
    Simulate concept presence based on genre characteristics.
    This is a placeholder until we have actual concept extraction data.
    """
    np.random.seed(42)
    n_samples = len(df)

    # Define genre-concept associations (based on art history knowledge)
    genre_concept_associations = {
        "Expressionism": [
            "expressive_brushstrokes",
            "gestural_marks",
            "vibrant_colors",
            "emotional_mood",
            "dynamic_energetic_mood",
            "thick_contour_lines",
            "aggressive_energy",
            "angular_fragmentation",
            "bold_outlines",
        ],
        "Impressionism": [
            "loose_visible_brushstrokes",
            "atmospheric_perspective",
            "diffused_light",
            "pastel_colors",
            "broken_sketchy_lines",
            "soft_rounded_forms",
            "peaceful_tranquil",
            "golden_hour_light",
            "blurred_soft_edges",
        ],
        "Baroque": [
            "chiaroscuro",
            "dramatic_shadows",
            "diagonal_composition",
            "rich_deep_colors",
            "religious_scenes",
            "dynamic_movement",
            "tense_dramatic",
            "hierarchical_scale",
            "flowing_organic_lines",
        ],
        "Renaissance": [
            "linear_perspective",
            "symmetrical_composition",
            "sfumato_technique",
            "religious_scenes",
            "portraits",
            "balanced_composition",
            "architectural_precision",
            "naturalistic_rendering",
            "smooth_blending",
        ],
        "Cubism": [
            "angular_fragmentation",
            "multiple_viewpoints",
            "geometric_shapes",
            "flattened_space",
            "muted_tones",
            "overlapping_forms",
            "abstract_forms",
            "compressed_space",
            "sharp_angular_forms",
        ],
        "Byzantine": [
            "gold_background",
            "religious_scenes",
            "hierarchical_scale",
            "flat_lighting",
            "symmetrical_composition",
            "symbolic_representation",
            "metallic_accents",
            "ornamental_borders",
            "static_composition",
        ],
        "Minimalism": [
            "minimal_detail",
            "geometric_shapes",
            "limited_palette",
            "flat_color_areas",
            "grid_structure",
            "pure_abstraction",
            "calm_serene_mood",
            "uniform_detail_level",
            "monochromatic",
        ],
        "Pop_Art": [
            "bright_saturated_colors",
            "bold_outlines",
            "flat_color_areas",
            "everyday_scenes",
            "repeating_motifs",
            "photo_quality_rendering",
            "celebratory_joyful",
            "high_contrast",
            "symbolic_objects",
        ],
    }

    # Anti-Expressionism markers (expanded list)
    anti_expressionism = [
        "smooth_surface",
        "polished_surface",
        "architectural_precision",
        "photo_quality_rendering",
        "delicate_precise_lines",
        "minimal_shadows",
        "even_illumination",
        "calm_serene_mood",
        "uniform_detail_level",
    ]

    # Create concept presence matrix
    concept_matrix = np.zeros((n_samples, len(ALL_CONCEPTS)))

    for idx, row in df.iterrows():
        # For each genre the image belongs to
        for genre in genre_columns:
            if row[genre] == 1 and genre in genre_concept_associations:
                # Add associated concepts
                for concept in genre_concept_associations[genre]:
                    if concept in ALL_CONCEPTS:
                        concept_idx = ALL_CONCEPTS.index(concept)
                        concept_matrix[idx, concept_idx] = 1

        # Add some random noise
        noise_concepts = np.random.choice(len(ALL_CONCEPTS), size=10, replace=False)
        concept_matrix[idx, noise_concepts] = np.random.choice(
            [0, 1], size=10, p=[0.7, 0.3]
        )

    return concept_matrix, anti_expressionism


def calculate_concept_discrimination(df, genre_columns, concept_matrix):
    """Calculate how well each concept discriminates between genres."""

    results = defaultdict(dict)

    # For each genre
    for genre in genre_columns:
        y = df[genre].values

        if y.sum() < 10:  # Skip genres with too few samples
            continue

        # Calculate chi2 scores for each concept
        chi2_scores, _ = chi2(concept_matrix, y)

        # Store top discriminative concepts for this genre
        top_indices = np.argsort(chi2_scores)[-10:][::-1]

        for idx in top_indices:
            concept = ALL_CONCEPTS[idx]
            score = chi2_scores[idx]
            results[genre][concept] = score

    return results


def find_balanced_concept_set(
    discrimination_results, anti_expressionism, target_size=40
):
    """Select a balanced set of concepts that discriminate well across all genres."""

    # Calculate overall importance for each concept
    concept_scores = defaultdict(float)
    concept_genres = defaultdict(set)

    for genre, concepts in discrimination_results.items():
        for concept, score in concepts.items():
            concept_scores[concept] += score
            concept_genres[concept].add(genre)

    # Boost anti-Expressionism markers
    for concept in anti_expressionism:
        if concept in concept_scores:
            concept_scores[concept] *= 1.5

    # Rank concepts by:
    # 1. Number of genres they help discriminate
    # 2. Total discrimination score
    concept_ranking = []
    for concept, score in concept_scores.items():
        n_genres = len(concept_genres[concept])
        concept_ranking.append((concept, score, n_genres))

    # Sort by number of genres first, then by score
    concept_ranking.sort(key=lambda x: (x[2], x[1]), reverse=True)

    # Select top concepts ensuring genre coverage
    selected_concepts = []
    genre_coverage = defaultdict(int)

    for concept, score, n_genres in concept_ranking:
        # Check which genres this concept helps with
        genres_helped = concept_genres[concept]

        # Prioritize concepts that help under-represented genres
        under_represented = [g for g in genres_helped if genre_coverage[g] < 3]

        if under_represented or len(selected_concepts) < target_size:
            selected_concepts.append(concept)
            for g in genres_helped:
                genre_coverage[g] += 1

        if len(selected_concepts) >= target_size:
            break

    # Manual override for balance - remove hyper-Expressionist concepts
    FORCE_OUT = {
        "expressive_brushstrokes",
        "gestural_marks",
        "aggressive_energy",
        "thick_contour_lines",
    }
    FORCE_IN = {
        "delicate_precise_lines",
        "polished_surface",
        "calm_serene_mood",
        "photo_quality_rendering",
    }

    # Remove forced-out concepts
    selected_concepts = [c for c in selected_concepts if c not in FORCE_OUT]

    # Add forced-in concepts if not already present
    for concept in FORCE_IN:
        if concept not in selected_concepts and concept in ALL_CONCEPTS:
            selected_concepts.append(concept)
            # Update genre coverage for tracking
            for genre in concept_genres.get(concept, []):
                genre_coverage[genre] += 1

    # Ensure we still have target_size concepts
    selected_concepts = selected_concepts[:target_size]

    return selected_concepts, genre_coverage


def analyze_concept_categories(selected_concepts):
    """Analyze distribution of selected concepts across categories."""
    category_counts = defaultdict(int)

    for concept in selected_concepts:
        for category, concepts in CONCEPT_CATEGORIES.items():
            if concept in concepts:
                category_counts[category] += 1
                break

    return category_counts


def main():
    print("🎨 CONCEPT DISCRIMINATION ANALYSIS")
    print("=" * 60)

    # Load data
    df, genre_columns, genre_counts = load_genre_data()

    # Simulate concept presence (replace with actual extraction data when available)
    print(f"\n🔄 Simulating concept presence for {len(ALL_CONCEPTS)} concepts...")
    concept_matrix, anti_expressionism = simulate_concept_presence(df, genre_columns)

    # Calculate discrimination scores
    print("\n📊 Calculating concept discrimination scores...")
    discrimination_results = calculate_concept_discrimination(
        df, genre_columns, concept_matrix
    )

    # Find balanced concept set
    print("\n🎯 Selecting balanced concept set...")
    selected_concepts, genre_coverage = find_balanced_concept_set(
        discrimination_results, anti_expressionism, target_size=40
    )

    # Results
    print(f"\n✅ SELECTED {len(selected_concepts)} MOST DISCRIMINATIVE CONCEPTS:")
    print("=" * 60)

    # Show concepts by category
    category_counts = analyze_concept_categories(selected_concepts)

    for category, count in sorted(
        category_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"\n{category.upper()} ({count} concepts):")
        category_concepts = [
            c for c in selected_concepts if c in CONCEPT_CATEGORIES[category]
        ]
        for concept in category_concepts:
            # Check if it's an anti-Expressionism marker
            marker = "⚖️ " if concept in anti_expressionism else "  "
            print(f"  {marker}{concept}")

    # Genre coverage analysis
    print("\n📈 GENRE COVERAGE:")
    for genre, count in sorted(
        genre_coverage.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {genre}: {count} discriminative concepts")

    # Expressionism check
    expr_concepts = [
        c
        for c in selected_concepts
        if c in discrimination_results.get("Expressionism", {})
    ]
    anti_expr_concepts = [c for c in selected_concepts if c in anti_expressionism]

    print(f"\n⚖️  EXPRESSIONISM BALANCE:")
    print(f"  Pro-Expressionism concepts: {len(expr_concepts)}")
    print(f"  Anti-Expressionism concepts: {len(anti_expr_concepts)}")

    # Save results
    output = {
        "selected_concepts": selected_concepts,
        "genre_coverage": dict(genre_coverage),
        "anti_expressionism_markers": [
            c for c in selected_concepts if c in anti_expressionism
        ],
        "category_distribution": dict(category_counts),
    }

    with open("model/cbm/discriminative_concepts.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n📁 Results saved to: model/cbm/discriminative_concepts.json")

    # Recommendation
    print("\n💡 RECOMMENDATION:")
    print(f"  Start with these {len(selected_concepts)} concepts for CBM training")
    print(
        f"  Estimated parameters: {1536 * len(selected_concepts):,} (vs {1536 * 161:,} for all)"
    )
    print(
        f"  This gives ~{len(df) // len(selected_concepts)} training samples per concept"
    )


if __name__ == "__main__":
    main()
