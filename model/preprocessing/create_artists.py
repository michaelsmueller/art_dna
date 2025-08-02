# model/create_artists.py
"""
Create artists.csv from resized image filenames
Extracts artist names and generates artist metadata
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import Counter


def extract_artist_name(filename):
    """
    Extract artist name from filename format: Artist_Name_123.jpg

    Args:
        filename: Image filename (e.g., "Vincent_van_Gogh_123.jpg")

    Returns:
        Clean artist name (e.g., "Vincent van Gogh")
    """
    # Remove .jpg extension
    name_without_ext = filename.replace(".jpg", "")

    # Split by underscore and remove the last part (number)
    parts = name_without_ext.split("_")

    # Remove the last part if it's a number
    if parts[-1].isdigit():
        parts = parts[:-1]

    # Join with spaces
    artist_name = " ".join(parts)

    return artist_name


def create_artists_csv():
    """
    Create artists.csv from image filenames in raw_data/resized/
    """
    print("🎨 Creating artists.csv from Image Filenames")
    print("=" * 45)

    # Get all image files
    resized_dir = Path("raw_data/resized")
    if not resized_dir.exists():
        print(f"❌ Directory not found: {resized_dir}")
        return None

    image_files = list(resized_dir.glob("*.jpg"))
    print(f"📂 Found {len(image_files)} images")

    if len(image_files) == 0:
        print("❌ No images found!")
        return None

    # Extract artist names from filenames
    artist_names = []
    filename_to_artist = {}

    for img_file in image_files:
        artist_name = extract_artist_name(img_file.name)
        artist_names.append(artist_name)
        filename_to_artist[img_file.name] = artist_name

    # Count images per artist
    artist_counts = Counter(artist_names)

    print(f"🎨 Found {len(artist_counts)} unique artists:")
    for artist, count in sorted(artist_counts.items()):
        print(f"   {artist:<25}: {count:>3} images")

    # Create artists DataFrame
    artists_data = []
    for i, (artist_name, image_count) in enumerate(sorted(artist_counts.items())):
        artists_data.append(
            {
                "id": i,
                "name": artist_name,
                "paintings": image_count,
                "years": "",  # Will be filled manually or from external source
                "genre": "",  # Will be determined from labeled_data.csv
                "nationality": "",  # Will be filled manually or from external source
                "bio": "",  # Will be filled manually or from external source
                "wikipedia": "",  # Will be filled manually or from external source
            }
        )

    artists_df = pd.DataFrame(artists_data)

    # Try to infer some genres from labeled_data.csv if it exists
    if Path("raw_data/labeled_data.csv").exists():
        print("\n🔍 Inferring genres from labeled_data.csv...")
        labeled_data = pd.read_csv("raw_data/labeled_data.csv")

        # Check columns
        print(f"📊 Labeled data columns: {list(labeled_data.columns)}")

        if "filename" in labeled_data.columns and "genre" in labeled_data.columns:
            # Map filenames to artists and collect genres
            artist_genres = {}

            for _, row in labeled_data.iterrows():
                filename = row["filename"]
                genre = row["genre"]

                if filename in filename_to_artist:
                    artist = filename_to_artist[filename]

                    if artist not in artist_genres:
                        artist_genres[artist] = set()

                    artist_genres[artist].add(genre)

            # Update artists DataFrame with genres
            for i, row in artists_df.iterrows():
                artist_name = row["name"]
                if artist_name in artist_genres:
                    genres = sorted(list(artist_genres[artist_name]))
                    artists_df.at[i, "genre"] = ",".join(genres)

            print(f"✅ Added genre information for artists")
        else:
            print("⚠️ Columns 'filename' or 'genre' not found in labeled_data.csv")
    else:
        print("⚠️ labeled_data.csv not found, genres will be empty")

    # Save artists.csv
    output_path = "raw_data/artists.csv"
    artists_df.to_csv(output_path, index=False)

    print(f"\n💾 Saved artists.csv to {output_path}")
    print(f"📊 Created {len(artists_df)} artist records")

    # Show sample of created data
    print(f"\n📋 Sample artists data:")
    print(artists_df.head())

    return artists_df


if __name__ == "__main__":
    artists_df = create_artists_csv()
    if artists_df is not None:
        print("\n✅ Artists CSV creation completed!")
    else:
        print("\n❌ Artists CSV creation failed!")
