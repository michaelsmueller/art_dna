"""
Extract concepts for full 8,027-image dataset.
Conservative-fast approach: ~6 hours, <$80 cost.
"""

import os
import json
import asyncio
import time
from datetime import datetime
import pandas as pd
from pathlib import Path

# Import our concept extractor
import sys

sys.path.append(".")
from model.cbm.concept_extraction_discriminative import DiscriminativeConceptExtractor


async def extract_full_concepts():
    """Extract concepts for full dataset with optimized conservative settings"""

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        exit(1)

    print("🚀 FULL DATASET CONCEPT EXTRACTION")
    print("=" * 50)

    # Load full dataset
    df = pd.read_csv("raw_data/final_df.csv")
    print(f"📊 Loaded {len(df)} total images")

    # Verify images exist
    print("🔍 Checking image availability...")
    existing_df = df[df["image_path"].apply(os.path.exists)]
    missing_count = len(df) - len(existing_df)

    if missing_count > 0:
        print(
            f"⚠️  {missing_count} images missing, proceeding with {len(existing_df)} available"
        )

    # Create extractor
    extractor = DiscriminativeConceptExtractor(api_key)

    # Optimized parameters for 6-hour target
    batch_size = 25  # Larger batches
    delay_between_images = 0.2  # Faster processing
    delay_between_batches = 1  # Quick batch transitions

    # Storage for results
    all_results = []
    processed_count = 0

    # Create output directory
    output_dir = Path("model/cbm/full_extraction")
    output_dir.mkdir(exist_ok=True)

    print(f"\n🔄 Starting extraction...")
    print(f"   Batch size: {batch_size}")
    print(f"   Image delay: {delay_between_images}s")
    print(f"   Batch delay: {delay_between_batches}s")
    print(f"   Total batches: {(len(existing_df) + batch_size - 1) // batch_size}")
    print(f"   Target rate: ~22 img/min")
    print(
        f"   Estimated time: ~{len(existing_df) / 22:.1f} minutes ({len(existing_df) / 22 / 60:.1f} hours)"
    )
    print(f"   Estimated cost: ~${len(existing_df) * 0.0098:.0f}")

    start_time = time.time()

    # Process in batches
    for batch_start in range(0, len(existing_df), batch_size):
        batch_end = min(batch_start + batch_size, len(existing_df))
        batch_df = existing_df.iloc[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(existing_df) + batch_size - 1) // batch_size

        print(f"\n📦 Batch {batch_num}/{total_batches} ({len(batch_df)} images)")

        batch_results = []

        for idx, row in batch_df.iterrows():
            img_path = row["image_path"]

            try:
                print(f"   Processing: {os.path.basename(img_path)}")

                # Extract concepts
                concept_scores = await extractor.extract_concepts_single(img_path)

                # Store result
                result = {
                    "image_path": img_path,
                    "artist_name": row["artist_name"],
                    "concepts": concept_scores,
                    "extraction_timestamp": datetime.now().isoformat(),
                }

                # Add genre labels for reference
                genre_columns = [
                    col
                    for col in existing_df.columns
                    if col not in ["image_path", "artist_name"]
                ]
                result["genres"] = {col: int(row[col]) for col in genre_columns}

                batch_results.append(result)
                processed_count += 1

                # Brief pause between images
                await asyncio.sleep(delay_between_images)

            except Exception as e:
                print(f"   ❌ Error processing {os.path.basename(img_path)}: {e}")
                continue

        # Add batch results to total
        all_results.extend(batch_results)

        # Save intermediate results (backup every 100 batches)
        if batch_num % 100 == 0 or batch_num == total_batches:
            checkpoint_file = output_dir / f"checkpoint_{batch_num:04d}.json"
            with open(checkpoint_file, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"   💾 Checkpoint saved: {checkpoint_file}")

        # Progress update
        elapsed = time.time() - start_time
        rate = processed_count / elapsed * 60 if elapsed > 0 else 0  # img/min
        remaining = len(existing_df) - processed_count
        eta_min = remaining / (rate / 60) if rate > 0 else 0

        print(f"   ✅ Processed {processed_count}/{len(existing_df)} images")
        print(f"   ⏱️  Rate: {rate:.1f} img/min, ETA: {eta_min/60:.1f} hours")

        # Pause between batches
        if batch_end < len(existing_df):
            await asyncio.sleep(delay_between_batches)

    # Save final results
    final_file = output_dir / "full_concepts_complete.json"
    with open(final_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Create summary
    total_time = time.time() - start_time
    summary = {
        "total_images_attempted": len(existing_df),
        "successfully_processed": len(all_results),
        "failed_count": len(existing_df) - len(all_results),
        "processing_time_hours": total_time / 3600,
        "average_rate_per_minute": len(all_results) / (total_time / 60),
        "extraction_date": datetime.now().isoformat(),
        "final_results_file": str(final_file),
        "concept_count": len(extractor.concepts),
        "estimated_cost_usd": len(all_results) * 0.0098,
    }

    summary_file = output_dir / "full_extraction_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n🎉 FULL EXTRACTION COMPLETE!")
    print(f"✅ Successfully processed: {len(all_results)}/{len(existing_df)} images")
    print(f"⏱️  Total time: {total_time/3600:.1f} hours")
    print(f"💰 Estimated cost: ${len(all_results) * 0.0098:.0f}")
    print(f"📁 Results saved to: {final_file}")
    print(f"📊 Summary saved to: {summary_file}")

    return all_results


def main():
    """Main async wrapper"""
    try:
        results = asyncio.run(extract_full_concepts())
        print(f"\n🚀 Ready for full-scale CBM training!")
        return results
    except KeyboardInterrupt:
        print(f"\n⚠️  Extraction interrupted by user")
        print(f"   Partial results saved in model/cbm/full_extraction/")


if __name__ == "__main__":
    main()
