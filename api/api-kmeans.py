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

# Initialize similarity service
similarity_service = SimilarityService()
similarity_service.initialize()

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

# Device for torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load PCA and KMeans models saved with joblib
model_path = "model"
kmeans = joblib.load(f"{model_path}/kmeans_model.joblib")

# Load precomputed style text embeddings and style prompts
emb_path = "embeddings"
style_text_embeddings = np.load("embeddings/style_text_embeddings.npy").astype(np.float32)

# Load CLIP model and preprocessing
model_clip, preprocess = clip.load("ViT-B/32", device=device)
model_clip.eval()

@app.post("/predict-kmeans")
def predict_kmeans(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict top-5 art styles from uploaded image using CLIP + KMeans.

    Returns the 5 most similar styles to the predicted cluster center.
    """
    try:
        # Read and preprocess uploaded image
        image_bytes = image.file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_embedding = model_clip.encode_image(image_tensor).cpu().float().numpy().astype(np.float32)

        print("DEBUG: image_embedding_np dtype:", image_embedding.dtype)
        print("DEBUG: kmeans.cluster_centers_ dtype:", kmeans.cluster_centers_.dtype)
        print("DEBUG: style_text_embeddings dtype:", style_text_embeddings.dtype)
        print("DEBUG: image_embedding_np shape:", image_embedding.shape)

        # Predict cluster using KMeans
        pred_cluster = kmeans.predict(image_embedding.astype(np.float64))[0]

        # Get best-match style from the hardcoded cluster_to_style mapping
        best_match_art_style = cluster_to_style.get(pred_cluster, "Unknown")

        # --- Compare cluster center to style prompts ---
        cluster_center = kmeans.cluster_centers_[pred_cluster].astype(np.float32)
        print("DEBUG: cluster_center dtype:", cluster_center.dtype)

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

        return {
            "best_match_art_style":  best_match_art_style,
            "top_5_closest_styles": top5_closest_styles

        }

    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e