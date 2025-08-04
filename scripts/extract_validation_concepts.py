"""
Extract concepts for 509-image validation sample.
Production-ready batch processing with reliability features.
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


async def extract_validation_concepts():
    """Extract concepts for validation sample with robust error handling"""

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        exit(1)

    print("🚀 VALIDATION CONCEPT EXTRACTION")
    print("=" * 50)

    # Load validation sample
    sample_df = pd.read_csv("model/cbm/validation_sample_500.csv")
    print(f"📊 Loaded {len(sample_df)} validation images")

    # Verify images exist
    missing_images = []
    for img_path in sample_df["image_path"]:
        if not os.path.exists(img_path):
            missing_images.append(img_path)

    if missing_images:
        print(f"⚠️  Found {len(missing_images)} missing images:")
        for img in missing_images[:5]:
            print(f"    {img}")
        if len(missing_images) > 5:
            print(f"    ... and {len(missing_images)-5} more")

        # Filter out missing images
        sample_df = sample_df[sample_df["image_path"].apply(os.path.exists)]
        print(f"📊 Proceeding with {len(sample_df)} available images")

    # Create extractor
    extractor = DiscriminativeConceptExtractor(api_key)

    # Batch processing parameters
    batch_size = 10  # Process 10 images at a time
    delay_between_batches = 2  # Seconds between batches

    # Storage for results
    all_results = []
    processed_count = 0

    # Create output directory
    output_dir = Path("model/cbm/validation_extraction")
    output_dir.mkdir(exist_ok=True)

    print(f"\n🔄 Starting extraction...")
    print(f"   Batch size: {batch_size}")
    print(f"   Total batches: {(len(sample_df) + batch_size - 1) // batch_size}")
    print(f"   Estimated time: ~{len(sample_df) * 3 / 60:.1f} minutes")

    start_time = time.time()

    # Process in batches
    for batch_start in range(0, len(sample_df), batch_size):
        batch_end = min(batch_start + batch_size, len(sample_df))
        batch_df = sample_df.iloc[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(sample_df) + batch_size - 1) // batch_size

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
                    for col in sample_df.columns
                    if col not in ["image_path", "artist_name"]
                ]
                result["genres"] = {col: int(row[col]) for col in genre_columns}

                batch_results.append(result)
                processed_count += 1

                # Brief pause between images
                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"   ❌ Error processing {os.path.basename(img_path)}: {e}")
                # Continue with next image
                continue

        # Add batch results to total
        all_results.extend(batch_results)

        # Save intermediate results (backup)
        intermediate_file = output_dir / f"batch_{batch_num:03d}_results.json"
        with open(intermediate_file, "w") as f:
            json.dump(batch_results, f, indent=2)

        # Progress update
        elapsed = time.time() - start_time
        rate = processed_count / elapsed if elapsed > 0 else 0
        remaining = len(sample_df) - processed_count
        eta = remaining / rate if rate > 0 else 0

        print(f"   ✅ Processed {processed_count}/{len(sample_df)} images")
        print(f"   ⏱️  Rate: {rate:.1f} img/min, ETA: {eta/60:.1f} min")

        # Pause between batches (rate limiting)
        if batch_end < len(sample_df):
            print(f"   ⏳ Waiting {delay_between_batches}s before next batch...")
            await asyncio.sleep(delay_between_batches)

    # Save final results
    final_file = output_dir / "validation_concepts_complete.json"
    with open(final_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Create summary
    total_time = time.time() - start_time
    summary = {
        "total_images": len(sample_df),
        "successfully_processed": len(all_results),
        "failed_count": len(sample_df) - len(all_results),
        "processing_time_minutes": total_time / 60,
        "average_rate_per_minute": len(all_results) / (total_time / 60),
        "extraction_date": datetime.now().isoformat(),
        "final_results_file": str(final_file),
        "concept_count": len(extractor.concepts),
    }

    summary_file = output_dir / "extraction_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n🎉 EXTRACTION COMPLETE!")
    print(f"✅ Successfully processed: {len(all_results)}/{len(sample_df)} images")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📁 Results saved to: {final_file}")
    print(f"📊 Summary saved to: {summary_file}")

    if len(all_results) < len(sample_df):
        failed_count = len(sample_df) - len(all_results)
        print(f"⚠️  {failed_count} images failed - check batch files for details")

    return all_results


def main():
    """Main async wrapper"""
    try:
        results = asyncio.run(extract_validation_concepts())
        print(f"\n🚀 Ready for validation-scale CBM training!")
        return results
    except KeyboardInterrupt:
        print(f"\n⚠️  Extraction interrupted by user")
        print(f"   Partial results saved in model/cbm/validation_extraction/")


if __name__ == "__main__":
    main()
