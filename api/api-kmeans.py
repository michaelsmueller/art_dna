"""
Art DNA API - updated version of FastAPI backend for art style classification.

Provides endpoints for:
- Image prediction using Kmeans and CLIP
- 5 similar images using Kmeans, PCA and CLIP
- Genre descriptions for educational context
"""

from typing import Any, Dict

import io
import os
import numpy as np
import joblib
import clip
import torch

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from api.descriptions import DESCRIPTIONS
from api.similarity import SimilarityService
from api.helper_functions import Helpers

helpers = Helpers()
helpers.initialize()

similarity_service = SimilarityService()
similarity_service.initialize()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Only mount static files for local development
if os.getenv("USE_GCS", "false").lower() != "true":
    from fastapi.staticfiles import StaticFiles

    app.mount("/static", StaticFiles(directory="."), name="static")
    print("📁 Static files mounted for local development")

# Device for torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load PCA and KMeans models saved with joblib
model_path = "model"
kmeans = joblib.load(f"{model_path}/kmeans_model.joblib")
pca = joblib.load(f"{model_path}/pca_model.joblib")

# Load precomputed images embeddings, and style names embeddings
emb_path = "embeddings"
style_text_embeddings = np.load("embeddings/style_text_embeddings.npy").astype(np.float32)
pca_embeddings = np.load("embeddings/pca_embeddings.npy").astype(np.float32)

# Load CLIP model and preprocessing
model_clip, preprocess = clip.load("ViT-B/32", device=device)
model_clip.eval()

# Style prompts - 18 classes
style_prompts = [
    'Abstractionism', 'Art Nouveau', 'Baroque',
    'Byzantine Art', 'Cubism', 'Expressionism',
    'Impressionism', 'Mannerism', 'Muralism',
    'Neoplasticism', 'Pop Art', 'Primitivism',
    'Realism', 'Renaissance', 'Romanticism',
    'Suprematism', 'Surrealism', 'Symbolism'
]

cluster_to_style = {
    0: "Cubism",
    1: "Expressionism",
    2: "Realism",
    3: "Pop Art",
    4: "Art Nouveau",
    5: "Surrealism",
    6: "Neoplasticism",
    7: "Symbolism",
    8: "Impressionism",
    9: "Suprematism",
    10: "Renaissance",
    11: "Primitivism",
    12: "Byzantine Art",
    13: "Baroque",
    14: "Abstractionism",
    15: "Mannerism",
    16: "Muralism",
    17: "Romanticism",
}

@app.get("/")
def root():
    """Health check endpoint"""
    return {"greeting": "Hello"}

@app.get("/describe")
def describe_genres(
    genres: str = Query(...), audience: str = Query("adult")
) -> Dict[str, Any]:
    """Get descriptions for one or more art genres"""
    genre_list = [g.strip() for g in genres.split(",")]

    # Validate audience
    if audience not in DESCRIPTIONS:
        raise HTTPException(status_code=400, detail="Audience must be 'adult' or 'kid'")

    # Validate genres against known class names
    invalid_genres = [g for g in genre_list if g not in style_prompts]
    if invalid_genres:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid genres: {', '.join(invalid_genres)}. Valid genres: {', '.join(style_prompts)}",
        )

    descriptions = [
        DESCRIPTIONS[audience].get(
            genre,
            {
                "genre": genre,
                "description": f"Description for {genre} coming soon!",
                "time_period": "TBD",
                "key_artists": [],
                "visual_elements": [],
                "philosophy": "Coming soon",
            },
        )
        for genre in genre_list
    ]

    return {"audience": audience, "descriptions": descriptions}


@app.post("/predict-kmeans-new")

def predict_kmeans_new(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict top-5 art styles and find 5 visually similar paintings using CLIP + PCA + KMeans.
    """
    try:
        #Read and preprocess the uploaded image
        image_bytes = image.file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_embedding = model_clip.encode_image(image_tensor).cpu().numpy().astype(np.float32)

        # Predict cluster using KMeans
        pred_cluster = kmeans.predict(image_embedding.astype(np.float64))[0]

        # Get best-match style from the hardcoded cluster_to_style mapping
        best_match_art_style = cluster_to_style.get(pred_cluster, "Unknown")

        cluster_center = kmeans.cluster_centers_[pred_cluster].astype(np.float32)

        #Compare cluster center to style names CLIP embeddings
        cluster_center = kmeans.cluster_centers_[pred_cluster].astype(np.float32)
        similarities = np.dot(style_text_embeddings, cluster_center.T)
        top5_idx = similarities.argsort()[::-1][:5]

        # Format top-5 predictions
        top5_closest_styles = [
            {
                "art_style": style_prompts[i],
                "similarity_score": round(float(similarities[i]), 2),
            }
            for i in top5_idx
        ]

        #PCA transform the image embedding
        image_pca = pca.transform(image_embedding.astype(np.float64)).astype(np.float32)

        #Restrict to images in the same cluster
        same_cluster_indices = np.where(kmeans.labels_ == pred_cluster)[0]

        #Use Helpers class funciton to find similar paintings within cluster
        similar_images = helpers.find_similar(
            query_embedding=image_pca[0],
            top_k=5,
            restrict_indices=same_cluster_indices
        )

        return {
            "best_match_art_style": best_match_art_style,
            "top_5_closest_styles": top5_closest_styles,
            "similar_images": similar_images
        }

    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

#This is the API root for finding 5 similar images with DeIT, as in the previous app version

@app.post("/similar")
def find_similar_artworks(
    image: UploadFile = File(...), top_k: int = Query(5, ge=1, le=20)
) -> Dict[str, Any]:
    """Find visually similar artworks using DeiT embeddings"""
    try:
        # Read and process uploaded image
        image_bytes = image.file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Extract embedding
        query_embedding = similarity_service.extract_embedding(pil_image)

        # Find similar images
        similar_images = similarity_service.find_similar(query_embedding, top_k)

        return {"similar_images": similar_images}

    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Similarity search failed: {e}"
        ) from e
