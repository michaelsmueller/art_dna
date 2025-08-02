"""
Similarity search service using DeiT embeddings.
"""

import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, AutoModel
from typing import List, Dict, Any


class SimilarityService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.feature_extractor = None
        self.embeddings = None
        self.image_paths = None
        self.artists_df = None

    def load_model(self):
        """Load DeiT model and preprocessor"""
        model_name = "facebook/deit-base-distilled-patch16-224"

        try:
            self.feature_extractor = AutoImageProcessor.from_pretrained(
                model_name, use_fast=True
            )
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()  # Set to evaluation mode
            print(f"✅ DeiT model loaded on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load DeiT model: {e}")

    def load_embeddings(self):
        """Load precomputed embeddings and image paths"""
        try:
            embeddings_path = "embeddings/deit_embeddings.npy"
            paths_path = "embeddings/deit_paths.npy"

            if not os.path.exists(embeddings_path) or not os.path.exists(paths_path):
                raise FileNotFoundError(f"Embedding files not found in embeddings/")

            self.embeddings = np.load(embeddings_path)
            self.image_paths = np.load(paths_path, allow_pickle=True)

            print(
                f"✅ Loaded {len(self.embeddings)} embeddings with shape {self.embeddings.shape}"
            )
            print(f"✅ Loaded {len(self.image_paths)} image paths")

        except Exception as e:
            raise RuntimeError(f"Failed to load embeddings: {e}")

    def load_artists_metadata(self):
        """Load artists.csv for metadata joining"""
        try:
            artists_path = "raw_data/artists.csv"

            if not os.path.exists(artists_path):
                raise FileNotFoundError(f"Artists metadata not found at {artists_path}")

            self.artists_df = pd.read_csv(artists_path)

            # Normalize artist names for joining (lowercase, handle spaces/underscores)
            self.artists_df["normalized_name"] = (
                self.artists_df["name"]
                .str.lower()
                .str.replace(" ", "_")
                .str.replace("-", "_")
            )

            print(f"✅ Loaded metadata for {len(self.artists_df)} artists")

        except Exception as e:
            raise RuntimeError(f"Failed to load artists metadata: {e}")

    def initialize(self):
        """Initialize all components at startup"""
        print("🚀 Initializing Similarity Service...")
        self.load_model()
        self.load_embeddings()
        self.load_artists_metadata()
        print("✅ Similarity Service ready!")

    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract embedding from uploaded image"""
        try:
            # Preprocess image (same as Kristina's approach)
            image = image.convert("RGB").resize((224, 224))

            # Use feature extractor for preprocessing
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Extract embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

            return embedding

        except Exception as e:
            raise RuntimeError(f"Failed to extract embedding: {e}")

    def find_similar(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find top-k similar images with metadata"""
        try:
            # Calculate cosine similarities
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]

            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                # Get image path and extract artist name from filename
                img_path = self.image_paths[idx]
                filename = os.path.basename(img_path)

                # Extract artist name from filename (e.g., "Claude_Monet_123.jpg" -> "claude_monet")
                artist_name_from_file = "_".join(filename.split("_")[:-1]).lower()

                # Find matching artist in metadata
                artist_row = self.artists_df[
                    self.artists_df["normalized_name"] == artist_name_from_file
                ]

                # Build result with metadata
                result = {
                    "filename": filename,
                    "image_url": self.generate_image_url(filename),
                    "similarity_score": float(round(similarities[idx], 4)),
                }

                # Add artist metadata if found
                if not artist_row.empty:
                    artist_data = artist_row.iloc[0]
                    result.update(
                        {
                            "artist_name": artist_data["name"],
                            "years": artist_data["years"],
                            "genre": artist_data["genre"],
                            "nationality": artist_data["nationality"],
                        }
                    )
                else:
                    result.update(
                        {
                            "artist_name": "Unknown",
                            "years": "Unknown",
                            "genre": "Unknown",
                            "nationality": "Unknown",
                        }
                    )

                results.append(result)

            return results

        except Exception as e:
            raise RuntimeError(f"Failed to find similar images: {e}")

    def generate_image_url(self, filename: str) -> str:
        """Generate image URL based on deployment environment"""
        use_gcs = os.getenv("USE_GCS", "false").lower() == "true"

        if use_gcs:
            # Production: GCS public URL
            bucket_name = "art-dna-ml-data"
            return f"https://storage.googleapis.com/{bucket_name}/raw_data/resized/{filename}"
        else:
            # Local: return absolute path to file
            return os.path.join(os.getcwd(), "raw_data", "resized", filename)
