# model/cbm/concept_extraction_discriminative.py
"""
Art concept extraction using discriminative concepts only.
Uses the 40 balanced concepts selected through discrimination analysis.
"""

import os
import json
import asyncio
from typing import List, Dict
from PIL import Image
import base64
from io import BytesIO
import anthropic
from tqdm import tqdm
from collections import defaultdict


# Load discriminative concepts
def load_discriminative_concepts():
    """Load the 40 selected discriminative concepts."""
    with open("model/cbm/discriminative_concepts.json", "r") as f:
        data = json.load(f)
    return data["selected_concepts"]


# Use only discriminative concepts
DISCRIMINATIVE_CONCEPTS = load_discriminative_concepts()

EXTRACTION_PROMPT = """Analyze this artwork and identify which visual concepts are present.

Use the following scoring scale:
- 0: Completely absent - no trace of this concept
- 0.3: Barely present - subtle hint or minimal presence  
- 0.7: Clearly present - obvious but not dominant feature
- 1: Dominant feature - one of the most prominent aspects

Visual concepts to evaluate (40 discriminative concepts):
{concepts_list}

Respond in JSON format like:
{{
  "concept_name": score,
  ...
}}

Focus on objective visual features, not interpretation. Be precise about what you can actually see. If uncertain between two scores, choose the lower one."""


class DiscriminativeConceptExtractor:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.concepts = DISCRIMINATIVE_CONCEPTS  # Only 40 concepts

    def image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 for API."""
        with Image.open(image_path) as img:
            # Resize if too large (max 1024px)
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Save to bytes
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)

            return base64.b64encode(buffer.read()).decode("utf-8")

    async def extract_concepts_single(self, image_path: str) -> Dict[str, float]:
        """Extract concepts from a single image."""
        try:
            # Convert image
            image_base64 = self.image_to_base64(image_path)

            # Format concepts list
            concepts_list = "\n".join([f"- {concept}" for concept in self.concepts])

            # Create message
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0,  # Deterministic
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": EXTRACTION_PROMPT.format(
                                    concepts_list=concepts_list
                                ),
                            },
                        ],
                    }
                ],
            )

            # Parse response
            response_text = message.content[0].text

            # Extract JSON from response
            import re

            json_match = re.search(r"\{[^}]+\}", response_text, re.DOTALL)
            if json_match:
                concepts_scores = json.loads(json_match.group())
            else:
                concepts_scores = json.loads(response_text)

            return concepts_scores

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return {}

    def extract_concepts_batch(self, image_paths: List[str], output_file: str):
        """Extract concepts from multiple images and save results."""
        results = []

        print(f"Extracting concepts from {len(image_paths)} images...")
        print(f"Using {len(self.concepts)} discriminative visual concepts")
        print(f"Anti-Expressionism markers: 5 concepts included")

        for image_path in tqdm(image_paths):
            # Run async extraction
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            concepts = loop.run_until_complete(self.extract_concepts_single(image_path))
            loop.close()

            if concepts:
                results.append({"image_path": image_path, "concepts": concepts})

            # Save incrementally
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            # Small delay to respect rate limits
            import time

            time.sleep(0.5)

        print(f"\nExtraction complete! Results saved to: {output_file}")

        # Analyze concept frequency
        self.analyze_concept_distribution(results)

        return results

    def analyze_concept_distribution(self, results: List[Dict]):
        """Analyze which concepts are most common/discriminative."""
        concept_counts = defaultdict(float)
        concept_presence = defaultdict(int)

        for result in results:
            for concept, score in result.get("concepts", {}).items():
                if score >= 0.7:  # Count as clearly present
                    concept_presence[concept] += 1
                concept_counts[concept] += score

        print("\nConcept Distribution Analysis:")
        print(f"Total images analyzed: {len(results)}")
        print("\nMost common concepts:")

        sorted_concepts = sorted(
            concept_presence.items(), key=lambda x: x[1], reverse=True
        )
        for concept, count in sorted_concepts[:10]:
            percentage = (count / len(results)) * 100
            print(f"  {concept}: {count} images ({percentage:.1f}%)")


if __name__ == "__main__":
    # Test with discriminative concepts
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        print("⚠️  Please set ANTHROPIC_API_KEY environment variable")
    else:
        print("🎯 Using 40 discriminative concepts")
        print("✅ Balanced for Expressionism (6 pro / 5 anti)")

        extractor = DiscriminativeConceptExtractor(api_key)

        # Quick test with one image
        import pandas as pd

        df = pd.read_csv("raw_data/final_df.csv")
        test_image = df.iloc[0]["image_path"]

        print(f"\nTest extraction on: {test_image}")
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(extractor.extract_concepts_single(test_image))
        loop.close()

        print(f"Extracted {len(result)} concept scores")
