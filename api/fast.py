"""
Art DNA API - FastAPI backend for art style classification.

Provides endpoints for:
- Image prediction using VGG16 model
- Genre descriptions for educational context
"""

from typing import Any, Dict

import io
import os
import numpy as np

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
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

# Load model and class names once at startup
model = load_model("model/art_style_classifier.keras", compile=False)
with open("model/class_names.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

# Initialize similarity service
similarity_service = SimilarityService()
similarity_service.initialize()


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
    invalid_genres = [g for g in genre_list if g not in class_names]
    if invalid_genres:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid genres: {', '.join(invalid_genres)}. Valid genres: {', '.join(class_names)}",
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


@app.post("/predict")
def predict(image: UploadFile = File(...)) -> Dict[str, Dict[str, float]]:
    """
    Predict art style from uploaded image.

    Returns probabilities for all 18 art genres.
    """
    try:
        # Read and decode the uploaded image
        image_bytes = image.file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_image = pil_image.resize((224, 224))

        # Preprocess for model
        array = np.array(pil_image) / 255.0
        array = np.expand_dims(array, axis=0)

        # Predict
        probs = model.predict(array)[0]
        predictions = {
            class_names[i]: float(round(probs[i], 4)) for i in range(len(class_names))
        }

        return {"predictions": predictions}

    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e


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
