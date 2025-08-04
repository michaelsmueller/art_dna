"""
Create stratified 500-image sample for validation-scale concept extraction.
Maintains genre balance while handling class imbalance.
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split


def analyze_genre_distribution(df):
    """Analyze current genre distribution"""
    genre_columns = [
        col for col in df.columns if col not in ["image_path", "artist_name"]
    ]

    print("📊 Current Genre Distribution:")
    genre_counts = {}

    for genre in genre_columns:
        count = df[genre].sum()
        percentage = (count / len(df)) * 100
        genre_counts[genre] = count
        print(f"  {genre:<15}: {count:4d} ({percentage:5.1f}%)")

    # Multi-label stats
    multi_label_count = (df[genre_columns].sum(axis=1) > 1).sum()
    print(
        f"\n📈 Multi-label images: {multi_label_count} ({multi_label_count/len(df)*100:.1f}%)"
    )

    return genre_counts


def create_stratified_sample(df, target_size=500, min_per_genre=5):
    """Create stratified sample maintaining genre proportions"""
    print(f"\n🎯 Creating stratified sample of {target_size} images...")

    genre_columns = [
        col for col in df.columns if col not in ["image_path", "artist_name"]
    ]
    genre_counts = {}

    # Calculate current proportions
    for genre in genre_columns:
        genre_counts[genre] = df[genre].sum()

    total_genre_instances = sum(genre_counts.values())

    # Calculate target samples per genre (proportional)
    target_per_genre = {}
    allocated_samples = 0

    for genre in genre_columns:
        proportion = genre_counts[genre] / total_genre_instances
        target = max(min_per_genre, int(target_size * proportion))
        target_per_genre[genre] = min(
            target, genre_counts[genre]
        )  # Can't exceed available
        allocated_samples += target_per_genre[genre]

    print(f"📋 Target samples per genre:")
    for genre, target in sorted(
        target_per_genre.items(), key=lambda x: x[1], reverse=True
    ):
        current = genre_counts[genre]
        print(f"  {genre:<15}: {target:3d} / {current:4d} available")

    # Sample from each genre
    selected_samples = []
    used_indices = set()

    for genre in genre_columns:
        target_count = target_per_genre[genre]

        # Get all images with this genre
        genre_images = df[df[genre] == 1]

        # Remove already selected images
        available_images = genre_images[~genre_images.index.isin(used_indices)]

        if len(available_images) >= target_count:
            # Random sample
            sampled = available_images.sample(n=target_count, random_state=42)
        else:
            # Take all available
            sampled = available_images
            print(
                f"  ⚠️  {genre}: Only {len(sampled)} available (wanted {target_count})"
            )

        selected_samples.append(sampled)
        used_indices.update(sampled.index.tolist())

    # Combine all selected samples
    sample_df = pd.concat(selected_samples).drop_duplicates()

    print(f"\n✅ Selected {len(sample_df)} unique images")

    return sample_df


def validate_sample(sample_df, original_df):
    """Validate the sample maintains good properties"""
    print(f"\n🔍 Sample Validation:")

    genre_columns = [
        col for col in sample_df.columns if col not in ["image_path", "artist_name"]
    ]

    # Genre distribution in sample
    print(f"📊 Sample Genre Distribution:")
    for genre in genre_columns:
        sample_count = sample_df[genre].sum()
        original_count = original_df[genre].sum()
        sample_pct = (sample_count / len(sample_df)) * 100
        original_pct = (original_count / len(original_df)) * 100

        print(
            f"  {genre:<15}: {sample_count:3d} ({sample_pct:5.1f}%) vs orig ({original_pct:5.1f}%)"
        )

    # Multi-label preservation
    sample_multi = (sample_df[genre_columns].sum(axis=1) > 1).sum()
    sample_multi_pct = (sample_multi / len(sample_df)) * 100

    original_multi = (original_df[genre_columns].sum(axis=1) > 1).sum()
    original_multi_pct = (original_multi / len(original_df)) * 100

    print(f"\n📈 Multi-label preservation:")
    print(f"  Sample: {sample_multi} ({sample_multi_pct:.1f}%)")
    print(f"  Original: {original_multi} ({original_multi_pct:.1f}%)")

    # Artist diversity
    sample_artists = sample_df["artist_name"].nunique()
    original_artists = original_df["artist_name"].nunique()

    print(f"\n🎨 Artist diversity:")
    print(f"  Sample: {sample_artists} unique artists")
    print(f"  Original: {original_artists} unique artists")
    print(f"  Coverage: {(sample_artists/original_artists)*100:.1f}%")


def main():
    print("🎯 VALIDATION SAMPLE CREATION")
    print("=" * 50)

    # Load full dataset
    print("📁 Loading full dataset...")
    df = pd.read_csv("raw_data/final_df.csv")
    print(f"✅ Loaded {len(df)} total images")

    # Analyze current distribution
    genre_counts = analyze_genre_distribution(df)

    # Create stratified sample
    sample_df = create_stratified_sample(df, target_size=500, min_per_genre=8)

    # Validate sample
    validate_sample(sample_df, df)

    # Save sample
    output_file = "model/cbm/validation_sample_500.csv"
    sample_df.to_csv(output_file, index=False)

    print(f"\n💾 Sample saved to: {output_file}")

    # Save metadata
    metadata = {
        "total_original": len(df),
        "total_sample": len(sample_df),
        "target_size": 500,
        "min_per_genre": 8,
        "genres": len(
            [col for col in df.columns if col not in ["image_path", "artist_name"]]
        ),
        "sample_file": output_file,
        "created_for": "validation_scale_concept_extraction",
    }

    with open("model/cbm/validation_sample_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n🎉 Ready for concept extraction!")
    print(f"📋 Next: Run concept extraction on {len(sample_df)} images")
    print(f"💰 Estimated cost: ~${len(sample_df) * 2 * 0.015:.1f} (with 2 runs)")


if __name__ == "__main__":
    main()
