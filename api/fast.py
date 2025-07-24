import io
from typing import Dict, Any
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from PIL import Image, UnidentifiedImageError

from api.descriptions import DESCRIPTIONS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and class names once at startup
model = load_model("model/art_style_classifier.keras")

with open("model/class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]


@app.get("/")
def root():
    return {"greeting": "Hello"}


@app.get("/describe")
def describe_genres(
    genres: str = Query(..., description="Comma-separated genre names"),
    audience: str = Query("adult", description="Target audience: 'adult' or 'kid'"),
) -> Dict[str, Any]:
    """Get descriptions for one or more art genres"""

    # Parse genres
    genre_list = [g.strip() for g in genres.split(",")]

    # Validate audience
    if audience not in ["adult", "kid"]:
        raise HTTPException(status_code=400, detail="Audience must be 'adult' or 'kid'")

    # Get descriptions
    descriptions = []
    for genre in genre_list:
        if genre in DESCRIPTIONS[audience]:
            descriptions.append(DESCRIPTIONS[audience][genre])
        else:
            # Return placeholder for missing genres
            descriptions.append(
                {
                    "genre": genre,
                    "description": f"Description for {genre} not available yet",
                    "time_period": "TBD",
                    "key_artists": [],
                    "visual_elements": [],
                    "philosophy": "Coming soon",
                }
            )

    return {"audience": audience, "descriptions": descriptions}


@app.post("/predict")
def predict(image: UploadFile = File(...)) -> Dict[str, Dict[str, float]]:
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
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
