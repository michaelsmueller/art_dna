"""
Resume full concept extraction from last checkpoint.
"""

import json
import os
from pathlib import Path
import pandas as pd


def find_last_checkpoint():
    """Find the most recent checkpoint file"""
    checkpoint_dir = Path("model/cbm/full_extraction")

    if not checkpoint_dir.exists():
        print("❌ No extraction directory found")
        return None, 0

    # Find all checkpoint files
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))

    if not checkpoints:
        print("❌ No checkpoint files found")
        return None, 0

    # Get the latest checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split("_")[1]))

    # Load and count processed images
    with open(latest_checkpoint, "r") as f:
        results = json.load(f)

    processed_paths = {r["image_path"] for r in results}

    print(f"📁 Found checkpoint: {latest_checkpoint}")
    print(f"✅ Already processed: {len(processed_paths)} images")

    return processed_paths, len(processed_paths)


def create_resume_dataset(processed_paths):
    """Create dataset of remaining images"""
    # Load full dataset
    df = pd.read_csv("raw_data/final_df.csv")
    existing_df = df[df["image_path"].apply(os.path.exists)]

    # Filter out already processed images
    remaining_df = existing_df[~existing_df["image_path"].isin(processed_paths)]

    print(f"📊 Original dataset: {len(existing_df)} images")
    print(f"⏭️  Remaining to process: {len(remaining_df)} images")

    # Save remaining dataset
    remaining_file = "model/cbm/remaining_images.csv"
    remaining_df.to_csv(remaining_file, index=False)

    print(f"💾 Remaining images saved to: {remaining_file}")

    return len(remaining_df)


def main():
    print("🔄 EXTRACTION RESUME HELPER")
    print("=" * 40)

    # Find last checkpoint
    processed_paths, processed_count = find_last_checkpoint()

    if processed_paths is None:
        print("❌ Cannot resume - no checkpoints found")
        return

    # Create resume dataset
    remaining_count = create_resume_dataset(processed_paths)

    if remaining_count == 0:
        print("🎉 Extraction already complete!")
        return

    # Calculate time/cost for remaining
    estimated_hours = remaining_count / 22 / 60  # 22 img/min
    estimated_cost = remaining_count * 0.0098

    print(f"\n📋 Resume Summary:")
    print(f"   Already completed: {processed_count} images")
    print(f"   Remaining: {remaining_count} images")
    print(f"   Estimated time: {estimated_hours:.1f} hours")
    print(f"   Estimated cost: ${estimated_cost:.0f}")

    print(f"\n🚀 To resume extraction:")
    print(
        f"   1. Modify extract_full_concepts.py to use 'model/cbm/remaining_images.csv'"
    )
    print(f"   2. Or create a new script that loads remaining_images.csv")
    print(f"   3. Results will be merged with existing checkpoints")


if __name__ == "__main__":
    main()
