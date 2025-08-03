#!/usr/bin/env python3
"""
Quick spot check script to test Grad-CAM API endpoints and visualize results locally.
"""

import requests
import base64
import json
from pathlib import Path
import os

# Configuration
API_BASE = "http://localhost:8000"
TEST_IMAGE = "raw_data/test_images/mona_lisa.jpg"  # Update this path as needed
OUTPUT_DIR = "gradcam_test_output"


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)


def upload_image_and_predict(image_path):
    """Upload image to predict_cbm and get session_id."""
    print(f"🔄 Uploading {image_path} for CBM prediction...")

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print(
            "Please place a test image in the current directory or update TEST_IMAGE path"
        )
        return None

    with open(image_path, "rb") as f:
        files = {"image": f}
        response = requests.post(f"{API_BASE}/predict_cbm", files=files)

    if response.status_code == 200:
        result = response.json()
        print(f"✅ CBM Prediction successful!")
        print(f"📍 Session ID: {result['session_id']}")
        print(f"🎨 Top predicted genres: {result['predicted_genres'][:3]}")
        print(f"🔍 Top concepts: {[c['name'] for c in result['concepts'][:3]]}")
        return result
    else:
        print(f"❌ Prediction failed: {response.status_code}")
        print(response.text)
        return None


def get_gradcam_image(session_id, endpoint_type, name):
    """Get grad-cam image from API and save locally."""
    endpoint = f"{API_BASE}/gradcam/{session_id}/{endpoint_type}/{name}"
    print(f"🔄 Getting {endpoint_type} grad-cam for: {name}")

    response = requests.get(endpoint)

    if response.status_code == 200:
        result = response.json()

        # Extract base64 data from data URI (format: "data:image/png;base64,{base64_data}")
        gradcam_image = result["gradcam_image"]
        if gradcam_image.startswith("data:image/png;base64,"):
            base64_data = gradcam_image.split(",", 1)[1]
        else:
            base64_data = gradcam_image

        # Decode base64 image
        image_data = base64.b64decode(base64_data)

        # Save image
        filename = f"{endpoint_type}_{name}_{session_id[:8]}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(image_data)

        score_info = result.get("probability", result.get("score", "N/A"))
        print(f"✅ Saved {filepath} (score: {score_info:.3f})")
        return filepath
    else:
        print(
            f"❌ Failed to get {endpoint_type} grad-cam for {name}: {response.status_code}"
        )
        return None


def main():
    """Run the spot check."""
    print("🧪 Starting Grad-CAM API Spot Check")
    print("=" * 50)

    ensure_output_dir()

    # Step 1: Upload image and get predictions
    prediction_result = upload_image_and_predict(TEST_IMAGE)
    if not prediction_result:
        return

    session_id = prediction_result["session_id"]

    print("\n" + "=" * 50)
    print("🎨 Testing Style Grad-CAMs")
    print("=" * 50)

    # Step 2: Test style grad-cams for top 3 predicted genres
    top_genres = prediction_result["predicted_genres"][:3]
    style_images = []

    for genre in top_genres:
        filepath = get_gradcam_image(session_id, "style", genre)
        if filepath:
            style_images.append(filepath)

    print("\n" + "=" * 50)
    print("🔍 Testing Concept Grad-CAMs")
    print("=" * 50)

    # Step 3: Test concept grad-cams for top 3 concepts
    top_concepts = [c["name"] for c in prediction_result["concepts"][:3]]
    concept_images = []

    for concept in top_concepts:
        filepath = get_gradcam_image(session_id, "concept", concept)
        if filepath:
            concept_images.append(filepath)

    # Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🎨 Style grad-cams saved: {len(style_images)}")
    print(f"🔍 Concept grad-cams saved: {len(concept_images)}")

    if style_images or concept_images:
        print(
            "\n✅ Spot check complete! Open the saved PNG files to view the grad-cam heatmaps:"
        )
        for img in style_images + concept_images:
            print(f"   📷 {img}")
        print(f"\n💡 You can open them with: open {OUTPUT_DIR}/*.png")
    else:
        print("\n❌ No images were saved successfully")


if __name__ == "__main__":
    # Check if FastAPI server is running
    try:
        response = requests.get(f"{API_BASE}/")
        print("🚀 FastAPI server is running")
    except requests.exceptions.ConnectionError:
        print("❌ FastAPI server not running. Start it with: make run-backend")
        exit(1)

    main()
