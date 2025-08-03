"""
Defines helper functions for images similarity predicitons.
"""
import os
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Any
import pandas as pd

class Helpers:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model = None
        #self.feature_extractor = None
        self.embeddings = None          # np.ndarray of all PCA embeddings
        self.image_paths = None         # List[str] paths or filenames
        self.artists_df = None          # pd.DataFrame with artist metadata

    def initialize(self):
        """Initialize all components at startup"""
        print("🚀 Initializing Helpers Service...")

        self.embeddings = np.load("embeddings/pca_embeddings.npy").astype(np.float32)
        self.load_artists_metadata()
        # Automatically list all filenames in resized folder
        resized_dir = os.path.join("raw_data", "resized")
        self.image_paths = [
            fname for fname in os.listdir(resized_dir)
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        print("✅ Helpers Service ready!")

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


    def find_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        # restrict_indices is an optional list or array of indices to restrict search to (e.g., images in the same cluster). If None, it compares with all images.
        restrict_indices: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:

        """Find top-k similar images with metadata, optionally restricted to subset indices."""

        try:
            #If restrict_indices is provided (e.g., same cluster), use only that subset
            if restrict_indices is not None:
                restricted_embeddings = self.embeddings[restrict_indices]
                restricted_paths = [self.image_paths[i] for i in restrict_indices]
            else:
                restricted_embeddings = self.embeddings
                restricted_paths = self.image_paths

            # Compute cosine similarity
            similarities = cosine_similarity([query_embedding], restricted_embeddings)[0]

            # Get top-k indices by similarity descending
            top_indices = similarities.argsort()[::-1][:top_k]

            results = []
            for rank, local_idx in enumerate(top_indices):
                filename = os.path.basename(restricted_paths[local_idx])

                # Build result dictionary
                result = {
                    "rank": rank + 1,
                    "filename": filename,
                    "similarity_score": float(round(similarities[local_idx], 4)),
                    "image_url": self.generate_image_url(filename),
                }

                # Add artist metadata if available
                if self.artists_df is not None:
                    # Extract artist normalized name from filename
                    artist_key = "_".join(filename.split("_")[:-1]).lower()
                    artist_row = self.artists_df[self.artists_df["normalized_name"] == artist_key]

                    if not artist_row.empty:
                        artist_data = artist_row.iloc[0]
                        result.update({
                            "artist_name": artist_data["name"],
                            "years": artist_data["years"],
                            "genre": artist_data["genre"],
                            "nationality": artist_data["nationality"],
                        })
                    else:
                        # Defaults if no artist match
                        result.update({
                            "artist_name": "Unknown",
                            "years": "Unknown",
                            "genre": "Unknown",
                            "nationality": "Unknown",
                        })

                results.append(result)

            return results

        except Exception as e:
            raise RuntimeError(f"Failed to find similar images: {e}")
