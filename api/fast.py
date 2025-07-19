from typing import Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np
import io

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
        predictions = {class_names[i]: float(round(probs[i], 4)) for i in range(len(class_names))}

        return {"predictions": predictions}

    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
