"""
PyTorch data loader for multi-label art style classification.
Consolidates all tested components into production-ready pipeline.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional


class ArtStyleDataset(Dataset):
    """PyTorch Dataset for multi-label art style classification."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        root_dir: str = "raw_data/resized",
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            dataframe: DataFrame with image_path and genre columns
            root_dir: Root directory for images
            transform: Optional transform to be applied
        """
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

        # Get genre columns from class_names.txt
        with open("model/class_names.txt", "r") as f:
            self.genre_columns = [line.strip() for line in f if line.strip()]

        # Verify all genre columns exist in dataframe
        missing_cols = set(self.genre_columns) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing genre columns in dataframe: {missing_cols}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]

        # Load image
        img_path = row["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.root_dir, os.path.basename(img_path))

        image = Image.open(img_path).convert("RGB")

        # Get multi-label targets
        labels = row[self.genre_columns].values.astype(np.float32)
        labels = torch.tensor(labels)

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, labels


def get_transforms(is_train: bool = True, image_size: int = 224) -> transforms.Compose:
    """Get transforms for training or validation."""

    if is_train:
        # Training transforms with augmentation
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    return transform


def stratified_multilabel_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified split for multi-label data using primary genre.
    """
    # Get genre columns
    with open("model/class_names.txt", "r") as f:
        genre_columns = [line.strip() for line in f if line.strip()]

    # Use primary genre (most common per image) for stratification
    df_copy = df.copy()
    df_copy["primary_genre"] = df_copy[genre_columns].idxmax(axis=1)

    # Perform stratified split
    train_df, test_df = train_test_split(
        df_copy,
        test_size=test_size,
        stratify=df_copy["primary_genre"],
        random_state=random_state,
    )

    # Drop temporary column
    train_df = train_df.drop("primary_genre", axis=1)
    test_df = test_df.drop("primary_genre", axis=1)

    return train_df, test_df


def get_data_loaders(
    csv_path: str = "raw_data/final_df.csv",
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
    test_split: float = 0.2,
    random_state: int = 42,
    image_size: int = 224,
    verbose: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    Uses nested splitting: 64% train, 16% val, 20% test.

    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    # Load the dataframe
    df = pd.read_csv(csv_path)

    # Nested split: First split train+val vs test (80/20)
    train_val_df, test_df = stratified_multilabel_split(
        df, test_size=test_split, random_state=random_state
    )

    # Second split: train vs val (80/20 of remaining = 64%/16% total)
    train_df, val_df = stratified_multilabel_split(
        train_val_df, test_size=val_split, random_state=random_state
    )

    if verbose:
        print(f"Dataset splits:")
        print(f"  Train: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")

    # Create transforms
    train_transform = get_transforms(is_train=True, image_size=image_size)
    val_transform = get_transforms(is_train=False, image_size=image_size)

    # Create datasets
    train_dataset = ArtStyleDataset(train_df, transform=train_transform)
    val_dataset = ArtStyleDataset(val_df, transform=val_transform)
    test_dataset = ArtStyleDataset(test_df, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def calculate_pos_weights(csv_path: str = "raw_data/final_df.csv") -> torch.Tensor:
    """
    Calculate positive class weights for handling imbalanced multi-label data.
    Caps weights at 5.0 to prevent gradient explosion.

    Returns:
        Tensor of positive weights for BCEWithLogitsLoss
    """
    df = pd.read_csv(csv_path)

    # Get genre columns
    with open("model/class_names.txt", "r") as f:
        genre_columns = [line.strip() for line in f if line.strip()]

    # Calculate pos_weight = neg_count / pos_count, capped at 5.0
    pos_weights = []
    for genre in genre_columns:
        pos_count = df[genre].sum()
        neg_count = len(df) - pos_count
        raw_weight = neg_count / pos_count if pos_count > 0 else 1.0
        capped_weight = min(raw_weight, 5.0)
        pos_weights.append(capped_weight)

    return torch.tensor(pos_weights, dtype=torch.float32)
