#!/usr/bin/env python3
"""
Analyze missing files in pca_paths.npy to understand what needs to be fixed.
"""

import numpy as np
import os
import unicodedata
from collections import Counter


def analyze_missing_files():
    # Load the paths
    paths_file = "embeddings/pca_paths.npy"
    print(f"Loading paths from {paths_file}...")
    paths = np.load(paths_file, allow_pickle=True)

    # Find missing files
    missing = []
    existing = []

    for p in paths:
        full_path = os.path.join("raw_data", "resized", str(p))
        if not os.path.exists(full_path):
            missing.append(str(p))
        else:
            existing.append(str(p))

    print(f"\n📊 Summary:")
    print(f"Total paths: {len(paths)}")
    print(f"Existing files: {len(existing)}")
    print(f"Missing files: {len(missing)}")
    print(f"Percentage missing: {len(missing)/len(paths)*100:.2f}%")

    if not missing:
        print("\n✅ All files exist!")
        return

    # Analyze missing files by artist
    print(f"\n🎨 Missing files by artist:")
    missing_artists = Counter()
    for m in missing:
        # Extract artist name (everything before the last underscore and number)
        parts = m.rsplit("_", 1)
        if len(parts) > 1:
            artist = parts[0]
            missing_artists[artist] += 1

    for artist, count in missing_artists.most_common():
        print(f"  {artist}: {count} files")

    # Show sample of missing files
    print(f"\n📁 First 10 missing files:")
    for m in missing[:10]:
        print(f"  {m}")

    # Check for encoding issues
    print(f"\n🔤 Checking for encoding issues...")
    sample = missing[0] if missing else None
    if sample:
        print(f"Sample file: {sample}")
        print(f"Bytes: {sample.encode('utf-8')}")
        print(f"Has special chars: {not sample.isascii()}")

        # Check for unicode normalization issues
        nfd = unicodedata.normalize("NFD", sample)
        nfc = unicodedata.normalize("NFC", sample)
        if nfd != sample or nfc != sample:
            print(f"NFD normalized: {nfd}")
            print(f"NFC normalized: {nfc}")

        # Check if it's the umlaut issue
        if "ü" in sample or "Ü" in sample:
            print(f"⚠️  Contains umlaut characters (ü/Ü)")
            # Try alternative spellings
            alternatives = [
                sample.replace("ü", "u").replace("Ü", "U"),
                sample.replace("ü", "ue").replace("Ü", "Ue"),
            ]
            print(f"\nChecking alternative spellings:")
            for alt in alternatives:
                alt_path = os.path.join("raw_data", "resized", alt)
                exists = os.path.exists(alt_path)
                print(f"  {alt}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")

    # Check what files actually exist in the directory for the problematic artist
    if missing_artists:
        top_artist = list(missing_artists.keys())[0]
        print(f"\n🔍 Checking what files exist for '{top_artist}':")

        # Get the base name without special characters
        base_name = top_artist.replace("ü", "u").replace("Ü", "U")

        # List actual files
        resized_dir = "raw_data/resized"
        matching_files = []
        for f in os.listdir(resized_dir):
            if base_name.lower() in f.lower():
                matching_files.append(f)

        if matching_files:
            print(f"Found {len(matching_files)} similar files:")
            for f in matching_files[:5]:
                print(f"  {f}")
        else:
            print(f"No files found matching '{base_name}'")

    # Suggest fix
    print(f"\n💡 Suggested fix:")
    if any("ü" in m or "Ü" in m for m in missing):
        print(
            "The issue is with umlaut characters (ü). The files in pca_paths.npy use 'ü'"
        )
        print("but the actual files might use 'u' or 'ue'. You can:")
        print("1. Fix pca_paths.npy to match actual filenames")
        print("2. Rename actual files to match pca_paths.npy")
        print("3. Remove these entries from the embeddings")


if __name__ == "__main__":
    analyze_missing_files()
