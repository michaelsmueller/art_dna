# model/create_final_df.py
"""
Create final_df.csv from artists.csv, labeled_data.csv, and resized images
Generates multi-label dataset for CBM training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.preprocessing import MultiLabelBinarizer


def create_final_df():
    """
    Create final_df.csv with format:
    image_path, artist_name, Abstractionism, Art Nouveau, ..., Symbolism
    """
    print("🏗️ Creating final_df.csv from Raw Data")
    print("=" * 40)

    # Load the labeled data
    labeled_data_path = "raw_data/labeled_data.csv"
    if not os.path.exists(labeled_data_path):
        print(f"❌ File not found: {labeled_data_path}")
        return None

    labeled_data = pd.read_csv(labeled_data_path)
    print(f"📂 Loaded labeled_data.csv: {len(labeled_data)} rows")
    print(f"📊 Columns: {list(labeled_data.columns)}")

    # Load artists data
    artists_path = "raw_data/artists.csv"
    if os.path.exists(artists_path):
        artists_df = pd.read_csv(artists_path)
        print(f"📂 Loaded artists.csv: {len(artists_df)} artists")
    else:
        print("⚠️ No artists.csv found")
        artists_df = None

    # Check available images
    resized_dir = Path("raw_data/resized")
    if not resized_dir.exists():
        print(f"❌ Directory not found: {resized_dir}")
        return None

    image_files = list(resized_dir.glob("*.jpg"))
    print(f"📂 Found {len(image_files)} images in raw_data/resized/")

    # Create filename to path mapping
    filename_to_path = {f.name: f"raw_data/resized/{f.name}" for f in image_files}

    # Check labeled_data structure
    print(f"\n📊 Sample labeled_data:")
    print(labeled_data.head())

    # Determine filename column
    filename_col = None
    for col in ["filename", "image", "file", "image_name"]:
        if col in labeled_data.columns:
            filename_col = col
            break

    if filename_col is None:
        print("❌ No filename column found in labeled_data.csv")
        print(f"Available columns: {list(labeled_data.columns)}")
        return None

    print(f"✅ Using filename column: '{filename_col}'")

    # Check genre column
    genre_col = None
    for col in ["genre", "style", "label", "class"]:
        if col in labeled_data.columns:
            genre_col = col
            break

    if genre_col is None:
        print("❌ No genre column found in labeled_data.csv")
        print(f"Available columns: {list(labeled_data.columns)}")
        return None

    print(f"✅ Using genre column: '{genre_col}'")

    # Filter to existing images only
    labeled_data = labeled_data[
        labeled_data[filename_col].isin(filename_to_path.keys())
    ]
    print(f"📊 After filtering to existing images: {len(labeled_data)} rows")

    if len(labeled_data) == 0:
        print("❌ No matching images found!")
        print(
            f"Sample filenames in labeled_data: {labeled_data[filename_col].head().tolist()}"
        )
        print(f"Sample image files: {[f.name for f in image_files[:5]]}")
        return None

    # Add full image paths
    labeled_data["image_path"] = labeled_data[filename_col].map(filename_to_path)

    # Extract artist names from filenames
    def extract_artist_name(filename):
        """Extract artist name from filename format: Artist_Name_123.jpg"""
        name_without_ext = filename.replace(".jpg", "")
        parts = name_without_ext.split("_")

        # Remove the last part if it's a number
        if parts[-1].isdigit():
            parts = parts[:-1]

        return " ".join(parts)

    labeled_data["artist_name"] = labeled_data[filename_col].apply(extract_artist_name)

    # Show sample data
    print(f"\n📊 Sample processed data:")
    sample_data = labeled_data[["image_path", "artist_name", genre_col]].head()
    print(sample_data)

    # Group by image to create multi-label dataset
    print(f"\n🔄 Creating multi-label dataset...")

    # Group multiple genres per image
    multi_label_df = (
        labeled_data.groupby(["image_path", "artist_name"])[genre_col]
        .apply(list)
        .reset_index()
    )
    print(f"📊 Grouped into {len(multi_label_df)} unique images")

    # Show sample multi-label data
    print(f"\n📊 Sample multi-label data:")
    for i in range(min(5, len(multi_label_df))):
        row = multi_label_df.iloc[i]
        print(f"   {row['artist_name']}: {row[genre_col]}")

    # Create binary encoding for genres using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(multi_label_df[genre_col])

    # Create genre columns DataFrame
    genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

    print(f"\n🎨 Found {len(mlb.classes_)} unique genres:")
    genre_counts = {}
    for genre in mlb.classes_:
        count = genre_df[genre].sum()
        genre_counts[genre] = count
        percentage = (count / len(multi_label_df)) * 100
        print(f"   {genre:<18}: {count:>4} images ({percentage:>5.1f}%)")

    # Combine image info with genre encodings
    final_df = pd.concat(
        [multi_label_df[["image_path", "artist_name"]], genre_df], axis=1
    )

    # Reorder columns to match expected format
    expected_genres = [
        "Abstractionism",
        "Art Nouveau",
        "Baroque",
        "Byzantine Art",
        "Cubism",
        "Expressionism",
        "Impressionism",
        "Mannerism",
        "Muralism",
        "Neoplasticism",
        "Pop Art",
        "Primitivism",
        "Realism",
        "Renaissance",
        "Romanticism",
        "Suprematism",
        "Surrealism",
        "Symbolism",
    ]

    # Check which expected genres we have
    available_genres = [g for g in expected_genres if g in genre_df.columns]
    missing_genres = [g for g in expected_genres if g not in genre_df.columns]

    if missing_genres:
        print(f"\n⚠️ Missing expected genres: {missing_genres}")
        # Add missing genres as zero columns
        for genre in missing_genres:
            final_df[genre] = 0

    # Reorder columns
    column_order = ["image_path", "artist_name"] + expected_genres
    final_df = final_df[column_order]

    # Save the final dataset
    output_path = "raw_data/final_df.csv"
    final_df.to_csv(output_path, index=False)

    print(f"\n💾 Saved final_df.csv to {output_path}")
    print(f"📊 Final dataset shape: {final_df.shape}")
    print(f"📊 Columns: {list(final_df.columns)}")

    # Show final statistics
    print(f"\n📈 Final Dataset Statistics:")
    print(f"   Total images: {len(final_df)}")
    print(f"   Unique artists: {final_df['artist_name'].nunique()}")
    print(f"   Average genres per image: {final_df.iloc[:, 2:].sum(axis=1).mean():.2f}")

    return final_df


if __name__ == "__main__":
    final_df = create_final_df()
    if final_df is not None:
        print("\n✅ final_df.csv creation completed!")

        # Show sample final data
        print(f"\n📋 Sample final_df:")
        print(final_df.head())
    else:
        print("\n❌ final_df.csv creation failed!")
