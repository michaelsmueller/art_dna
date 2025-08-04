"""
Concept-aware dataset for CBM training.
Extends the basic art dataset to include concept labels alongside style labels.
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ConceptArtDataset(Dataset):
    """
    Dataset that loads images with both style labels and concept labels.
    """

    def __init__(
        self,
        csv_path: str,
        concept_json_path: str = "model/cbm/data/full_extraction/full_concepts_complete.json",
        transform: Optional[transforms.Compose] = None,
        indices: Optional[np.ndarray] = None,
    ):
        """
        Initialize concept-aware dataset.

        Args:
            csv_path: Path to CSV with image paths and style labels
            concept_json_path: Path to JSON with extracted concept labels
            transform: Image transformations
            indices: Subset indices for train/val/test splits
        """
        # Load style labels from CSV
        self.df = pd.read_csv(csv_path)

        # Apply subset if specified
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)

        # Load concept data
        print(f"📊 Loading concepts from {concept_json_path}...")
        with open(concept_json_path, "r") as f:
            concept_data = json.load(f)

        # Create lookup dict for faster access
        self.concept_lookup = {
            item["image_path"]: item["concepts"] for item in concept_data
        }

        # Load concept names in order
        concept_names_path = (
            Path(concept_json_path).parent.parent / "final_concepts.json"
        )
        with open(concept_names_path, "r") as f:
            concept_info = json.load(f)
        self.concept_names = concept_info["selected_concepts"]
        self.n_concepts = len(self.concept_names)

        # Get style columns (all except image_path and artist_name)
        self.style_columns = [
            col for col in self.df.columns if col not in ["image_path", "artist_name"]
        ]
        self.n_classes = len(self.style_columns)

        self.transform = transform

        # Verify data alignment
        self._verify_data_alignment()

    def _verify_data_alignment(self):
        """Verify that all images have concept labels."""
        missing_concepts = []
        for idx, row in self.df.iterrows():
            if row["image_path"] not in self.concept_lookup:
                missing_concepts.append(row["image_path"])

        if missing_concepts:
            print(f"⚠️  Warning: {len(missing_concepts)} images missing concept labels")
            print(f"   First 5: {missing_concepts[:5]}")
        else:
            print(f"✅ All {len(self.df)} images have concept labels")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get image, style labels, and concept labels.

        Returns:
            image: Transformed image tensor
            style_labels: Multi-hot style labels [n_classes]
            concept_labels: Concept scores [n_concepts]
        """
        # Get image path and style labels
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        # Convert style labels to numeric, handling any non-numeric values
        style_values = []
        for col in self.style_columns:
            val = row[col]
            # Convert to float, default to 0 if not convertible
            try:
                style_values.append(float(val))
            except (ValueError, TypeError):
                style_values.append(0.0)

        style_labels = torch.tensor(style_values, dtype=torch.float32)

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Get concept labels
        if img_path in self.concept_lookup:
            concepts_dict = self.concept_lookup[img_path]
            # Convert to ordered array based on concept names
            concept_scores = []
            for concept_name in self.concept_names:
                score = concepts_dict.get(concept_name, 0.0)
                concept_scores.append(score)
            concept_labels = torch.tensor(concept_scores, dtype=torch.float32)
        else:
            # Fallback: zeros if concepts missing (shouldn't happen after verification)
            concept_labels = torch.zeros(self.n_concepts, dtype=torch.float32)

        return image, style_labels, concept_labels


def get_concept_data_loaders(
    csv_path: str = "raw_data/final_df.csv",
    concept_json_path: str = "model/cbm/data/full_extraction/full_concepts_complete.json",
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.2,
    num_workers: int = 0,
    pin_memory: bool = False,  # Set to False for M1 Mac MPS compatibility
    augment_train: bool = True,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train/val/test data loaders with concept labels.

    Returns:
        Dictionary with 'train', 'val', 'test' data loaders
    """
    # Load full dataset to get indices
    full_df = pd.read_csv(csv_path)
    n_samples = len(full_df)

    # Create stratified splits (simplified - could use sklearn for better stratification)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    # Calculate split points
    val_size = int(n_samples * val_split)
    test_size = int(n_samples * test_split)
    train_size = n_samples - val_size - test_size

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Define transforms
    if augment_train:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    val_test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = ConceptArtDataset(
        csv_path, concept_json_path, train_transform, train_indices
    )
    val_dataset = ConceptArtDataset(
        csv_path, concept_json_path, val_test_transform, val_indices
    )
    test_dataset = ConceptArtDataset(
        csv_path, concept_json_path, val_test_transform, test_indices
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"\n📊 Dataset Statistics:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")
    print(f"  Classes: {train_dataset.n_classes}")
    print(f"  Concepts: {train_dataset.n_concepts}")

    return {"train": train_loader, "val": val_loader, "test": test_loader}


if __name__ == "__main__":
    # Quick test
    print("🧪 Testing Concept Dataset")
    print("=" * 40)

    # Create a small loader
    loaders = get_concept_data_loaders(batch_size=2, num_workers=0)
    train_loader = loaders["train"]

    # Get one batch
    images, style_labels, concept_labels = next(iter(train_loader))

    print(f"\n✅ Successfully loaded batch:")
    print(f"  Images: {images.shape}")
    print(f"  Style labels: {style_labels.shape} (sum: {style_labels.sum(dim=1)})")
    print(f"  Concept labels: {concept_labels.shape}")
    print(f"  Active concepts per image: {(concept_labels > 0.5).sum(dim=1).float()}")
    print(
        f"  Concept value range: [{concept_labels.min():.2f}, {concept_labels.max():.2f}]"
    )
