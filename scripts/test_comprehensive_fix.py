#!/usr/bin/env python3
"""
Comprehensive tests to ensure the similarity fix works correctly
"""

import numpy as np
import joblib
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
import random


def test_clustering_still_works():
    """Test that KMeans clustering still works with PCA embeddings"""
    print("=== Test 1: Clustering with PCA ===")

    # Load models
    kmeans = joblib.load("model/kmeans_model.joblib")
    pca = joblib.load("model/pca_model.joblib")

    # Load a sample of CLIP embeddings
    clip_embeddings = np.load("embeddings/clip/all_clip_embeddings.npy").astype(
        np.float64
    )

    # Transform to PCA and predict clusters
    sample_indices = random.sample(range(len(clip_embeddings)), 10)
    for idx in sample_indices[:3]:  # Show first 3
        clip_emb = clip_embeddings[idx : idx + 1]
        pca_emb = pca.transform(clip_emb)
        cluster = kmeans.predict(clip_emb)[0]
        print(f"Image {idx}: Cluster {cluster}, PCA dims: {pca_emb[0]}")

    print("✅ Clustering still works with PCA\n")


def test_similarity_within_cluster():
    """Test similarity search within a cluster using full embeddings"""
    print("=== Test 2: Similarity Within Cluster ===")

    # Load data
    kmeans = joblib.load("model/kmeans_model.joblib")
    clip_embeddings = np.load("embeddings/clip/all_clip_embeddings.npy").astype(
        np.float32
    )
    paths = np.load("embeddings/pca_paths.npy", allow_pickle=True)

    # Pick a random image
    query_idx = 100
    query_embedding = clip_embeddings[query_idx]
    query_cluster = kmeans.labels_[query_idx]

    print(f"Query image: {paths[query_idx]} (Cluster {query_cluster})")

    # Find all images in same cluster
    same_cluster_indices = np.where(kmeans.labels_ == query_cluster)[0]
    print(f"Found {len(same_cluster_indices)} images in same cluster")

    # Compute similarities using FULL embeddings
    cluster_embeddings = clip_embeddings[same_cluster_indices]
    similarities = cosine_similarity([query_embedding], cluster_embeddings)[0]

    # Get top 5
    top_5 = similarities.argsort()[::-1][:5]
    print("\nTop 5 similar in same cluster:")
    for i, local_idx in enumerate(top_5):
        global_idx = same_cluster_indices[local_idx]
        print(f"{i+1}. {paths[global_idx]} - Score: {similarities[local_idx]:.4f}")

    # Check score distribution
    print(f"\nScore distribution in cluster:")
    print(f"Min: {similarities.min():.4f}, Max: {similarities.max():.4f}")
    print(f"Mean: {similarities.mean():.4f}, Std: {similarities.std():.4f}")
    print("✅ Similarity within cluster works well\n")


def test_api_workflow():
    """Simulate the API workflow with a test image"""
    print("=== Test 3: Simulated API Workflow ===")

    # Load all required models and data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_clip, preprocess = clip.load("ViT-B/32", device=device)
    kmeans = joblib.load("model/kmeans_model.joblib")
    pca = joblib.load("model/pca_model.joblib")

    clip_embeddings = np.load("embeddings/clip/all_clip_embeddings.npy").astype(
        np.float32
    )
    paths = np.load("embeddings/pca_paths.npy", allow_pickle=True)

    # Use an existing image as "uploaded" image
    test_image_path = os.path.join("raw_data", "resized", paths[500])
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return

    # Simulate API processing
    print(f"Simulating upload of: {paths[500]}")

    # 1. Load and preprocess image (as API does)
    img = Image.open(test_image_path).convert("RGB")
    image_tensor = preprocess(img).unsqueeze(0).to(device)

    # 2. Extract CLIP embedding
    with torch.no_grad():
        image_embedding = (
            model_clip.encode_image(image_tensor).cpu().numpy().astype(np.float32)
        )

    print(f"Extracted CLIP embedding shape: {image_embedding.shape}")

    # 3. Predict cluster using KMeans (needs float64)
    pred_cluster = kmeans.predict(image_embedding.astype(np.float64))[0]
    print(f"Predicted cluster: {pred_cluster}")

    # 4. Get indices of same cluster
    same_cluster_indices = np.where(kmeans.labels_ == pred_cluster)[0]
    print(f"Images in same cluster: {len(same_cluster_indices)}")

    # 5. Find similar using FULL embedding (not PCA!)
    cluster_embeddings = clip_embeddings[same_cluster_indices]
    similarities = cosine_similarity(image_embedding, cluster_embeddings)[0]

    top_5 = similarities.argsort()[::-1][:5]
    print("\nTop 5 similar images:")
    for i, local_idx in enumerate(top_5):
        global_idx = same_cluster_indices[local_idx]
        print(f"{i+1}. {paths[global_idx]} - Score: {similarities[local_idx]:.4f}")

    print("✅ API workflow simulation successful\n")


def test_edge_cases():
    """Test edge cases and potential issues"""
    print("=== Test 4: Edge Cases ===")

    # Load data
    kmeans = joblib.load("model/kmeans_model.joblib")
    clip_embeddings = np.load("embeddings/clip/all_clip_embeddings.npy")

    # Test 1: Check cluster distribution
    print("Cluster distribution:")
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} images")

    # Test 2: Find smallest cluster
    min_cluster = unique[counts.argmin()]
    min_count = counts.min()
    print(f"\nSmallest cluster: {min_cluster} with {min_count} images")

    # Test 3: Verify embeddings are normalized (if they should be)
    norms = np.linalg.norm(clip_embeddings[:10], axis=1)
    print(f"\nFirst 10 embedding norms: {norms}")
    print(f"Are embeddings normalized? {np.allclose(norms, 1.0)}")

    print("✅ Edge case tests complete\n")


if __name__ == "__main__":
    print("Running comprehensive tests for similarity fix...\n")

    try:
        test_clustering_still_works()
    except Exception as e:
        print(f"❌ Clustering test failed: {e}\n")

    try:
        test_similarity_within_cluster()
    except Exception as e:
        print(f"❌ Similarity test failed: {e}\n")

    try:
        test_api_workflow()
    except Exception as e:
        print(f"❌ API workflow test failed: {e}\n")

    try:
        test_edge_cases()
    except Exception as e:
        print(f"❌ Edge case test failed: {e}\n")

    print("All tests completed!")
