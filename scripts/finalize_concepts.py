"""
Finalize concept list by removing problematic concepts identified in pilot analysis.
Creates the final 37-concept vocabulary for production extraction.
"""

import json


def main():
    # Load current discriminative concepts
    with open("model/cbm/discriminative_concepts.json", "r") as f:
        data = json.load(f)

    current_concepts = data["selected_concepts"]
    anti_expressionism_markers = data["anti_expressionism_markers"]

    print("🔧 FINALIZING CONCEPT VOCABULARY")
    print("=" * 50)
    print(f"Current concepts: {len(current_concepts)}")

    # Concepts to remove based on pilot analysis
    concepts_to_remove = {
        "blurred_soft_edges",  # 100% presence - too generic
        "dramatic_shadows",  # 0.980 correlation with chiaroscuro
        "peaceful_tranquil",  # 0.955 correlation with calm_serene_mood
    }

    print(f"\nRemoving {len(concepts_to_remove)} problematic concepts:")
    for concept in concepts_to_remove:
        if concept in current_concepts:
            print(f"  ❌ {concept}")
        else:
            print(f"  ⚠️  {concept} (not found)")

    # Create final list
    final_concepts = [c for c in current_concepts if c not in concepts_to_remove]

    # Update anti-expressionism markers (remove any that were removed)
    final_anti_expressionism = [
        c for c in anti_expressionism_markers if c not in concepts_to_remove
    ]

    print(f"\nFinal concepts: {len(final_concepts)}")
    print(f"Anti-Expressionism markers: {len(final_anti_expressionism)}")

    # Verify we kept the important ones
    kept_important = []
    if "chiaroscuro" in final_concepts:
        kept_important.append("chiaroscuro (kept over dramatic_shadows)")
    if "calm_serene_mood" in final_concepts:
        kept_important.append("calm_serene_mood (kept over peaceful_tranquil)")

    if kept_important:
        print(f"\n✅ Kept important concepts:")
        for concept in kept_important:
            print(f"  ✓ {concept}")

    # Save final list
    final_data = {
        "selected_concepts": sorted(final_concepts),  # Keep alphabetical
        "anti_expressionism_markers": sorted(final_anti_expressionism),
        "removed_concepts": sorted(list(concepts_to_remove)),
        "removal_reasons": {
            "blurred_soft_edges": "100% presence - too generic",
            "dramatic_shadows": "0.980 correlation with chiaroscuro",
            "peaceful_tranquil": "0.955 correlation with calm_serene_mood",
        },
        "pilot_analysis_date": "2025-01-18",
        "final_count": len(final_concepts),
    }

    with open("model/cbm/final_concepts.json", "w") as f:
        json.dump(final_data, f, indent=2)

    print(f"\n📁 Final concept vocabulary saved to: model/cbm/final_concepts.json")

    # Show some sample concepts
    print(f"\n📝 Sample final concepts:")
    for i, concept in enumerate(sorted(final_concepts)[:10]):
        print(f"  {i+1:2d}. {concept}")
    if len(final_concepts) > 10:
        print(f"  ... and {len(final_concepts) - 10} more")

    print(
        f"\n🎯 Ready for production concept extraction with {len(final_concepts)} concepts!"
    )


if __name__ == "__main__":
    main()
