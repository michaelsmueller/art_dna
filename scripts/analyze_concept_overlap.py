#!/usr/bin/env python3
"""
Analyze concept activation correlations and spatial overlap in Grad-CAM heatmaps.
"""

import requests
import base64
import json
import numpy as np
from pathlib import Path
import os
from PIL import Image
import io
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
API_BASE = "http://localhost:8000"
TEST_IMAGES = [
    "raw_data/test_images/0andy-warhol-Marilyn-Monroe-the-shot-series2.jpg",
    "raw_data/test_images/pablo-picasso-guitar-and-mandolin.jpg",
    # Add more test images if available
]
OUTPUT_DIR = "concept_overlap_analysis"


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)


def get_image_predictions(image_path):
    """Get CBM predictions for an image."""
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        return None

    print(f"🔄 Analyzing {os.path.basename(image_path)}...")

    with open(image_path, "rb") as f:
        files = {"image": f}
        response = requests.post(f"{API_BASE}/predict_cbm", files=files)

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Got predictions - Session: {result['session_id'][:8]}...")
        return result
    else:
        print(f"❌ Prediction failed: {response.status_code}")
        return None


def get_gradcam_heatmap_array(session_id, concept_name):
    """Get grad-cam heatmap as numpy array."""
    endpoint = f"{API_BASE}/gradcam/{session_id}/concept/{concept_name}"
    response = requests.get(endpoint)

    if response.status_code == 200:
        result = response.json()

        # Extract base64 data
        gradcam_image = result["gradcam_image"]
        if gradcam_image.startswith("data:image/png;base64,"):
            base64_data = gradcam_image.split(",", 1)[1]
        else:
            base64_data = gradcam_image

        # Decode and convert to numpy array
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data))

        # Convert to numpy array and get just one channel (they should be identical for heatmaps)
        heatmap_array = np.array(image)
        if len(heatmap_array.shape) == 3:
            # Use the red channel (heatmaps are usually in red/yellow scale)
            heatmap_array = heatmap_array[:, :, 0]

        return heatmap_array, result.get("score", 0)
    else:
        print(f"❌ Failed to get concept grad-cam for {concept_name}")
        return None, 0


def compute_spatial_overlap(heatmap1, heatmap2, threshold=0.5):
    """Compute spatial overlap between two heatmaps."""
    # Normalize heatmaps to 0-1 range
    h1_norm = (heatmap1 - heatmap1.min()) / (heatmap1.max() - heatmap1.min() + 1e-8)
    h2_norm = (heatmap2 - heatmap2.min()) / (heatmap2.max() - heatmap2.min() + 1e-8)

    # Create binary masks for high-activation regions
    mask1 = h1_norm > threshold
    mask2 = h2_norm > threshold

    # Compute overlap metrics
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)

    # Jaccard similarity (IoU)
    jaccard = intersection / union if union > 0 else 0

    # Correlation between raw heatmap values
    correlation = np.corrcoef(h1_norm.flatten(), h2_norm.flatten())[0, 1]

    # Cosine similarity
    cos_sim = cosine_similarity(
        h1_norm.flatten().reshape(1, -1), h2_norm.flatten().reshape(1, -1)
    )[0, 0]

    return {
        "jaccard": jaccard,
        "correlation": correlation,
        "cosine_similarity": cos_sim,
        "intersection_pixels": intersection,
        "union_pixels": union,
    }


def analyze_concept_correlations(prediction_results):
    """Analyze correlations between concept activation scores."""
    print("\n📊 Analyzing concept activation correlations...")

    # Collect concept scores across all images
    all_concept_scores = {}
    concept_names = []

    for img_result in prediction_results:
        if img_result is None:
            continue

        for concept in img_result["concepts"]:
            name = concept["name"]
            score = concept["score"]

            if name not in all_concept_scores:
                all_concept_scores[name] = []
                if name not in concept_names:
                    concept_names.append(name)

            all_concept_scores[name].append(score)

    # Convert to matrix
    concept_matrix = []
    valid_concept_names = []

    for name in concept_names:
        if len(all_concept_scores[name]) == len(prediction_results):
            concept_matrix.append(all_concept_scores[name])
            valid_concept_names.append(name)

    concept_matrix = np.array(concept_matrix).T  # Images x Concepts

    # Compute correlation matrix
    concept_corr_matrix = np.corrcoef(concept_matrix.T)

    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(concept_corr_matrix, dtype=bool))
    sns.heatmap(
        concept_corr_matrix,
        annot=True,
        fmt=".2f",
        xticklabels=valid_concept_names,
        yticklabels=valid_concept_names,
        cmap="RdBu_r",
        center=0,
        mask=mask,
        square=True,
    )
    plt.title("Concept Activation Score Correlations")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/concept_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Find highly correlated concept pairs
    high_corr_pairs = []
    n = len(valid_concept_names)

    for i in range(n):
        for j in range(i + 1, n):
            corr = concept_corr_matrix[i, j]
            if abs(corr) > 0.7:  # High correlation threshold
                high_corr_pairs.append(
                    (valid_concept_names[i], valid_concept_names[j], corr)
                )

    print(
        f"🔍 Found {len(high_corr_pairs)} highly correlated concept pairs (|r| > 0.7):"
    )
    for concept1, concept2, corr in sorted(
        high_corr_pairs, key=lambda x: abs(x[2]), reverse=True
    ):
        print(f"   📎 {concept1} ↔ {concept2}: r = {corr:.3f}")

    return concept_corr_matrix, valid_concept_names, high_corr_pairs


def analyze_spatial_overlap_for_image(session_id, concepts, image_name):
    """Analyze spatial overlap between concept heatmaps for one image."""
    print(f"\n🎨 Analyzing spatial overlap for {image_name}...")

    # Get heatmaps for top concepts
    concept_heatmaps = {}
    concept_scores = {}

    for concept in concepts[:8]:  # Analyze top 8 concepts
        concept_name = concept["name"]
        heatmap, score = get_gradcam_heatmap_array(session_id, concept_name)
        if heatmap is not None:
            concept_heatmaps[concept_name] = heatmap
            concept_scores[concept_name] = score

    # Compute pairwise spatial overlaps
    concept_names = list(concept_heatmaps.keys())
    n_concepts = len(concept_names)

    overlap_matrix = np.zeros((n_concepts, n_concepts))
    correlation_matrix = np.zeros((n_concepts, n_concepts))

    print(f"🔄 Computing {n_concepts}x{n_concepts} pairwise overlaps...")

    for i, concept1 in enumerate(concept_names):
        for j, concept2 in enumerate(concept_names):
            if i == j:
                overlap_matrix[i, j] = 1.0
                correlation_matrix[i, j] = 1.0
            else:
                overlap_metrics = compute_spatial_overlap(
                    concept_heatmaps[concept1], concept_heatmaps[concept2]
                )
                overlap_matrix[i, j] = overlap_metrics["jaccard"]
                correlation_matrix[i, j] = overlap_metrics["correlation"]

    # Create spatial overlap heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Jaccard similarity (spatial overlap)
    im1 = ax1.imshow(overlap_matrix, cmap="Reds", vmin=0, vmax=1)
    ax1.set_title(f"Spatial Overlap (Jaccard) - {image_name}")
    ax1.set_xticks(range(n_concepts))
    ax1.set_yticks(range(n_concepts))
    ax1.set_xticklabels(concept_names, rotation=45, ha="right")
    ax1.set_yticklabels(concept_names)

    # Add text annotations
    for i in range(n_concepts):
        for j in range(n_concepts):
            ax1.text(
                j,
                i,
                f"{overlap_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if overlap_matrix[i, j] > 0.5 else "black",
            )

    plt.colorbar(im1, ax=ax1)

    # Correlation between heatmaps
    im2 = ax2.imshow(correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_title(f"Heatmap Correlations - {image_name}")
    ax2.set_xticks(range(n_concepts))
    ax2.set_yticks(range(n_concepts))
    ax2.set_xticklabels(concept_names, rotation=45, ha="right")
    ax2.set_yticklabels(concept_names)

    # Add text annotations
    for i in range(n_concepts):
        for j in range(n_concepts):
            ax2.text(
                j,
                i,
                f"{correlation_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black",
            )

    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()

    # Save plot
    safe_image_name = (
        os.path.basename(image_name).replace(".jpg", "").replace(".png", "")
    )
    plt.savefig(
        f"{OUTPUT_DIR}/spatial_overlap_{safe_image_name}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Report high overlaps
    high_overlap_pairs = []
    for i in range(n_concepts):
        for j in range(i + 1, n_concepts):
            jaccard = overlap_matrix[i, j]
            corr = correlation_matrix[i, j]
            if jaccard > 0.6 or abs(corr) > 0.8:  # High overlap threshold
                high_overlap_pairs.append(
                    (concept_names[i], concept_names[j], jaccard, corr)
                )

    print(f"🔍 High spatial overlap pairs for {image_name}:")
    for c1, c2, jaccard, corr in sorted(
        high_overlap_pairs, key=lambda x: x[2], reverse=True
    ):
        print(f"   📍 {c1} ↔ {c2}: Jaccard={jaccard:.3f}, Corr={corr:.3f}")

    return overlap_matrix, correlation_matrix, concept_names


def main():
    """Run the concept overlap analysis."""
    print("🔍 Starting Concept Overlap Analysis")
    print("=" * 60)

    ensure_output_dir()

    # Step 1: Get predictions for all test images
    prediction_results = []
    valid_images = []

    for image_path in TEST_IMAGES:
        result = get_image_predictions(image_path)
        prediction_results.append(result)
        if result:
            valid_images.append(image_path)

    # Filter out None results
    prediction_results = [r for r in prediction_results if r is not None]

    if not prediction_results:
        print("❌ No valid predictions obtained")
        return

    # Step 2: Analyze concept activation correlations
    concept_corr_matrix, concept_names, high_corr_pairs = analyze_concept_correlations(
        prediction_results
    )

    # Step 3: Analyze spatial overlap for each image
    for i, (result, image_path) in enumerate(zip(prediction_results, valid_images)):
        session_id = result["session_id"]
        concepts = result["concepts"]
        image_name = os.path.basename(image_path)

        overlap_matrix, correlation_matrix, spatial_concept_names = (
            analyze_spatial_overlap_for_image(session_id, concepts, image_name)
        )

    # Summary
    print("\n" + "=" * 60)
    print("📋 ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🖼️  Images analyzed: {len(prediction_results)}")
    print(f"🔍 Concepts analyzed: {len(concept_names) if concept_names else 'N/A'}")
    print(f"📎 High correlation pairs found: {len(high_corr_pairs)}")
    print(f"📊 Saved correlation heatmap: concept_correlations.png")
    print(f"🎨 Saved spatial overlap plots for each image")

    print(f"\n💡 View results: open {OUTPUT_DIR}/*.png")


if __name__ == "__main__":
    # Check if FastAPI server is running
    try:
        response = requests.get(f"{API_BASE}/")
        print("🚀 FastAPI server is running")
    except requests.exceptions.ConnectionError:
        print("❌ FastAPI server not running. Start it with: make run-backend")
        exit(1)

    main()
