#!/usr/bin/env python3
"""
Patch missing concept extraction for failed images.
"""

import os
import json
import asyncio
from pathlib import Path
import sys

# Add project root to path
sys.path.append(".")

from model.cbm.concept_extraction_discriminative import DiscriminativeConceptExtractor


async def patch_missing_extraction():
    """Extract concepts for the failed image and patch the results."""

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        return False

    print("🔧 PATCHING MISSING CONCEPT EXTRACTION")
    print("=" * 50)

    # Failed image
    failed_image = "raw_data/resized/Edgar_Degas_486.jpg"

    # Check if image exists
    if not Path(failed_image).exists():
        print(f"❌ Image not found: {failed_image}")
        return False

    print(f"📸 Extracting concepts for: {failed_image}")

    # Initialize extractor
    extractor = DiscriminativeConceptExtractor(api_key)

    # Extract concepts for the failed image
    try:
        concepts = await extractor.extract_concepts_single(failed_image)

        # Format result to match batch extraction format
        result = {"image_path": failed_image, "concepts": concepts}

        print(f"✅ Successfully extracted {len(result['concepts'])} concepts")

        # Load full results
        results_file = Path("model/cbm/full_extraction/full_concepts_complete.json")
        with open(results_file, "r") as f:
            all_results = json.load(f)

        # Find and patch the failed entry
        patched = False
        for i, entry in enumerate(all_results):
            if entry["image_path"] == failed_image:
                print(f"🔧 Patching entry {i+1}")
                all_results[i] = result
                patched = True
                break

        if not patched:
            print("❌ Failed entry not found in results - this shouldn't happen!")
            return False

        # Save patched results
        backup_file = results_file.with_suffix(".backup.json")
        print(f"💾 Creating backup: {backup_file}")

        # Create backup
        with open(backup_file, "w") as f:
            json.dump(all_results, f, indent=2)

        # Save patched version
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"✅ Successfully patched {results_file}")
        print(f"📋 Backup saved to: {backup_file}")

        return True

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(patch_missing_extraction())
    if success:
        print("\n🎉 Patch completed successfully!")
        print("💡 Run verification script again to confirm fix")
    else:
        print("\n❌ Patch failed")
        exit(1)
