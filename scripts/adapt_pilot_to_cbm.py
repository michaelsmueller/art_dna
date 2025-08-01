"""
Convert pilot concept extraction results to CBM pipeline format.
"""

import json


def main():
    print("🔄 Converting pilot data to CBM format...")

    # Load pilot results
    with open("model/cbm/pilot_results.json", "r") as f:
        pilot_data = json.load(f)

    # Load final concepts
    with open("model/cbm/final_concepts.json", "r") as f:
        final_concepts = json.load(f)["selected_concepts"]

    # Convert to CBM expected format
    cbm_format = []

    # Use run1 data (run2 is similar, just different LLM responses)
    for image_path, concept_scores in pilot_data["run1"].items():
        # Filter to only final concepts
        filtered_concepts = {}
        for concept in final_concepts:
            score = concept_scores.get(concept, 0)
            # Convert 4-level to binary for CBM
            filtered_concepts[concept] = 1 if score >= 0.7 else 0

        cbm_format.append({"image_path": image_path, "concepts": filtered_concepts})

    # Save in CBM format
    with open("model/cbm/pilot_concepts_cbm.json", "w") as f:
        json.dump(cbm_format, f, indent=2)

    print(f"✅ Converted {len(cbm_format)} samples")
    print(f"📁 Saved to: model/cbm/pilot_concepts_cbm.json")
    print(f"🎯 Ready to run CBM pipeline!")


if __name__ == "__main__":
    main()
