#!/usr/bin/env python3
"""
Reconstruct the filename ordering for PCA embeddings from the CLIP batch files.
This ensures the filenames match the order of embeddings in pca_embeddings.npy
"""

import os
import numpy as np
import glob


def reconstruct_pca_filenames():
    """
    Read the CLIP filename batches in order and save as pca_paths.npy
    """

    # Get all filename batch files, sorted by batch number
    clip_dir = "embeddings/clip"
    filename_files = sorted(
        glob.glob(os.path.join(clip_dir, "clip_filenames_batch_*.txt"))
    )

    print(f"Found {len(filename_files)} batch files")

    # Read all filenames in order
    all_filenames = []
    for fname in filename_files:
        print(f"Reading {os.path.basename(fname)}")
        with open(fname) as f:
            batch_filenames = [line.strip() for line in f if line.strip()]
            all_filenames.extend(batch_filenames)
            print(f"  Added {len(batch_filenames)} filenames")

    print(f"\nTotal filenames: {len(all_filenames)}")
    print(f"First 5: {all_filenames[:5]}")
    print(f"Last 5: {all_filenames[-5:]}")

    # Save as numpy array
    output_path = "embeddings/pca_paths.npy"
    np.save(output_path, all_filenames)
    print(f"\n✅ Saved {len(all_filenames)} filenames to {output_path}")

    # Verify counts match
    pca_embeddings = np.load("embeddings/pca_embeddings.npy")
    print(f"\nVerification:")
    print(f"PCA embeddings shape: {pca_embeddings.shape}")
    print(f"Number of filenames: {len(all_filenames)}")

    if pca_embeddings.shape[0] != len(all_filenames):
        print("⚠️  WARNING: Mismatch in counts!")
        print(f"Embeddings: {pca_embeddings.shape[0]}, Filenames: {len(all_filenames)}")
    else:
        print("✅ Counts match perfectly!")

    # Also check against all_clip_embeddings if available
    all_clip_path = os.path.join(clip_dir, "all_clip_embeddings.npy")
    if os.path.exists(all_clip_path):
        all_clip = np.load(all_clip_path)
        print(f"\nAll CLIP embeddings shape: {all_clip.shape}")
        if all_clip.shape[0] == len(all_filenames):
            print("✅ All CLIP embeddings count also matches!")


if __name__ == "__main__":
    reconstruct_pca_filenames()
