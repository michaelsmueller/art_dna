#!/usr/bin/env python3
"""
Fix the pca_paths.npy file by removing (1), (2), etc. suffixes that don't exist.
This is a one-time data cleanup to fix the source of the problem.
"""

import numpy as np
import re
import os
from datetime import datetime


def fix_pca_paths():
    # Load the current paths
    paths_file = "embeddings/pca_paths.npy"
    backup_file = (
        f"embeddings/pca_paths_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
    )

    print(f"Loading paths from {paths_file}...")
    paths = np.load(paths_file, allow_pickle=True)
    original_count = len(paths)

    # Create backup
    print(f"Creating backup at {backup_file}...")
    np.save(backup_file, paths)

    # Clean the paths
    cleaned_paths = []
    fixed_count = 0

    for path in paths:
        original_path = str(path)
        # Remove (1), (2), etc. suffixes
        cleaned_path = re.sub(r"\s*\(\d+\)\.jpg$", ".jpg", original_path.strip())

        if cleaned_path != original_path:
            print(f"Fixed: {original_path} -> {cleaned_path}")
            fixed_count += 1

        cleaned_paths.append(cleaned_path)

    # Convert back to numpy array with same dtype
    cleaned_paths_array = np.array(cleaned_paths, dtype=paths.dtype)

    # Verify all cleaned paths exist
    missing_files = []
    for path in cleaned_paths:
        full_path = os.path.join("raw_data", "resized", path)
        if not os.path.exists(full_path):
            missing_files.append(path)

    if missing_files:
        print(
            f"\nWarning: {len(missing_files)} files still don't exist after cleaning:"
        )
        for f in missing_files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")

    # Save the cleaned paths
    print(f"\nSaving cleaned paths to {paths_file}...")
    np.save(paths_file, cleaned_paths_array)

    print(f"\n✅ Fixed {fixed_count} paths out of {original_count} total paths")
    print(f"✅ Backup saved to {backup_file}")
    print(f"✅ pca_paths.npy has been updated with cleaned filenames")

    # Verify the fix
    print("\nVerifying the fix...")
    reloaded = np.load(paths_file, allow_pickle=True)
    bad_paths = [
        p for p in reloaded if "(1)" in str(p) or "(2)" in str(p) or "(3)" in str(p)
    ]
    if bad_paths:
        print(f"⚠️  Still found {len(bad_paths)} paths with (n) suffixes")
    else:
        print("✅ No more paths with (n) suffixes!")


if __name__ == "__main__":
    fix_pca_paths()
