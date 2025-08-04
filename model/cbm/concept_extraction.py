"""
Art concept extraction using Claude 4 Sonnet.
Extracts visual concepts that distinguish between art styles.
"""

import os
import json
import asyncio
from typing import List, Dict, Set
from PIL import Image
import base64
from io import BytesIO
import anthropic
from tqdm import tqdm
from collections import defaultdict


# Target visual concepts based on art style characteristics
CONCEPT_CATEGORIES = {
    "composition": [
        "asymmetric_balance",
        "centered_focal_point",
        "crowded_composition",
        "diagonal_composition",
        "golden_ratio",
        "grid_structure",
        "horizontal_emphasis",
        "prominent_negative_space",
        "radial_composition",
        "rule_of_thirds",
        "scattered_elements",
        "symmetrical_composition",
        "triangular_composition",
        "vertical_emphasis",
    ],
    "brushwork_and_line": [
        "broken_sketchy_lines",
        "calligraphic_strokes",
        "crosshatching",
        "delicate_precise_lines",
        "expressive_brushstrokes",
        "flat_color_areas",
        "fluid_sinuous_lines",
        "gestural_marks",
        "impasto_texture",
        "loose_visible_brushstrokes",
        "pointillist_dots",
        "sfumato_technique",
        "smooth_blending",
        "thick_contour_lines",
        "thin_delicate_lines",
    ],
    "color": [
        "analogous_colors",
        "black_and_white",
        "bright_saturated_colors",
        "complementary_colors",
        "cool_color_palette",
        "earth_tones",
        "gold_background",
        "high_contrast",
        "limited_palette",
        "low_contrast",
        "metallic_accents",
        "monochromatic",
        "muted_tones",
        "pastel_colors",
        "rich_deep_colors",
        "subdued_palette",
        "vibrant_colors",
        "warm_color_palette",
    ],
    "subject_matter": [
        "abstract_forms",
        "allegorical_content",
        "animals",
        "architectural_elements",
        "cityscapes",
        "everyday_scenes",
        "fantasy_elements",
        "geometric_shapes",
        "human_figures",
        "industrial_subjects",
        "landscape",
        "mythological_themes",
        "narrative_scene",
        "patterns_ornamental",
        "portraits",
        "religious_scenes",
        "seascapes",
        "still_life",
        "symbolic_objects",
    ],
    "perspective_and_space": [
        "aerial_perspective",
        "atmospheric_perspective",
        "compressed_space",
        "deep_recession",
        "distorted_perspective",
        "flattened_space",
        "foreshortening",
        "isometric_view",
        "linear_perspective",
        "multiple_viewpoints",
        "overlapping_forms",
        "shallow_depth",
        "spatial_ambiguity",
    ],
    "lighting": [
        "artificial_light_source",
        "backlighting",
        "chiaroscuro",
        "diffused_light",
        "dramatic_shadows",
        "even_illumination",
        "flat_lighting",
        "glowing_light",
        "golden_hour_light",
        "harsh_directional_light",
        "minimal_shadows",
        "reflected_light",
        "spotlight_effect",
    ],
    "texture_and_surface": [
        "architectural_precision",
        "blurred_soft_edges",
        "collage_elements",
        "cracked_surface",
        "detailed_rendering",
        "fabric_textures",
        "glass_transparency",
        "glossy_surface",
        "grainy_texture",
        "hard_crisp_edges",
        "matte_surface",
        "metallic_surfaces",
        "organic_textures",
        "patterned_surfaces",
        "photographic_precision",
        "polished_surface",
        "rough_texture",
        "simplified_forms",
        "smooth_surface",
        "textured_impasto",
        "weathered_patina",
    ],
    "visual_elements": [
        "angular_fragmentation",
        "biomorphic_shapes",
        "bold_outlines",
        "ornamental_borders",
        "dreamlike_imagery",
        "dynamic_movement",
        "elongated_proportions",
        "floating_elements",
        "flowing_organic_lines",
        "geometric_patterns",
        "hierarchical_scale",
        "naturalistic_rendering",
        "optical_effects",
        "primitive_simplified_forms",
        "repeating_motifs",
        "sharp_angular_forms",
        "soft_rounded_forms",
        "static_composition",
        "stylized_forms",
        "swirling_patterns",
    ],
    "abstraction_level": [
        "complete_realism",
        "figurative_abstraction",
        "high_realism",
        "moderate_stylization",
        "non_representational",
        "partial_abstraction",
        "photo_quality_rendering",
        "pure_abstraction",
        "semi_abstract",
        "symbolic_representation",
    ],
    "emotional_mood": [
        "aggressive_energy",
        "calm_serene_mood",
        "celebratory_joyful",
        "contemplative_mood",
        "dark_brooding",
        "dynamic_energetic_mood",
        "melancholic_mood",
        "mysterious_atmosphere",
        "peaceful_tranquil",
        "tense_dramatic",
        "whimsical_playful",
    ],
    "detail_density": [
        "densely_packed_detail",
        "intricate_detail",
        "minimal_detail",
        "moderate_detail",
        "selective_detail",
        "uniform_detail_level",
        "varying_detail_levels",
    ],
}

# Flatten all concepts
ALL_CONCEPTS = sorted(
    [concept for category in CONCEPT_CATEGORIES.values() for concept in category]
)

EXTRACTION_PROMPT = """Analyze this artwork and identify which visual concepts are present.

Use the following scoring scale:
- 0: Completely absent - no trace of this concept
- 0.3: Barely present - subtle hint or minimal presence  
- 0.7: Clearly present - obvious but not dominant feature
- 1: Dominant feature - one of the most prominent aspects

Visual concepts to evaluate (alphabetically sorted):
{concepts_list}

Respond in JSON format like:
{{
  "concept_name": score,
  ...
}}

Focus on objective visual features, not interpretation. Be precise about what you can actually see. If uncertain between two scores, choose the lower one."""


class ConceptExtractor:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.concepts = sorted(ALL_CONCEPTS)  # Ensure alphabetical order

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
        print(f"Using {len(self.concepts)} visual concepts")

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
                if score > 0.5:  # Count as present
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

        print("\nLeast common concepts:")
        for concept, count in sorted_concepts[-10:]:
            percentage = (count / len(results)) * 100
            print(f"  {concept}: {count} images ({percentage:.1f}%)")


def create_test_batch():
    """Create a small test batch for concept extraction."""
    import pandas as pd

    # Load the dataset
    df = pd.read_csv("raw_data/final_df.csv")

    # Get 2-3 images per style for testing
    test_images = []

    # Get genre columns
    genre_columns = [
        col for col in df.columns if col not in ["image_path", "artist_name"]
    ]

    for genre in genre_columns[:6]:  # Just first 6 genres for quick test
        # Get images for this genre
        genre_df = df[df[genre] == 1]
        if len(genre_df) > 0:
            # Take 2 samples
            samples = genre_df.sample(n=min(2, len(genre_df)), random_state=42)
            test_images.extend(samples["image_path"].tolist())

    print(f"Test batch created with {len(test_images)} images")
    return test_images


if __name__ == "__main__":
    # Quick test - update with your API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        print("⚠️  Please set ANTHROPIC_API_KEY environment variable")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
    else:
        extractor = ConceptExtractor(api_key)

        # Create small test batch
        test_images = create_test_batch()

        # Extract concepts
        output_file = "model/cbm/test_concepts.json"
        results = extractor.extract_concepts_batch(
            test_images[:5], output_file
        )  # Just 5 for quick test
