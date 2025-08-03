"""
Full-scale Concept Bottleneck Model (CBM) for art style classification.
Uses EfficientNet-B3 backbone with concept and style heads for interpretable predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3
from typing import Optional, Tuple, Dict, Any
import json
import os


class ConceptBottleneckModel(nn.Module):
    """
    Concept Bottleneck Model for interpretable art style classification.

    Architecture: Image → EfficientNet-B3 → Concepts → Styles
    Supports staged training: concept-only, style-only, or joint training.
    """

    def __init__(
        self,
        n_concepts: int = 37,
        n_classes: int = 18,
        backbone_weights: str = "IMAGENET1K_V1",
        concept_dropout: float = 0.3,
        style_dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        """
        Initialize CBM model.

        Args:
            n_concepts: Number of visual concepts
            n_classes: Number of art style classes
            backbone_weights: EfficientNet-B3 weights to use
            concept_dropout: Dropout rate for concept head
            style_dropout: Dropout rate for style head
            freeze_backbone: Whether to freeze backbone during training
        """
        super().__init__()

        self.n_concepts = n_concepts
        self.n_classes = n_classes

        # Load EfficientNet-B3 backbone
        self.backbone = efficientnet_b3(weights=backbone_weights)

        # EfficientNet-B3 has 1536 output features
        backbone_features = 1536

        # Replace classifier with identity to get raw features
        self.backbone.classifier = nn.Identity()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Concept prediction head
        self.concept_head = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.ReLU(),
            nn.Dropout(concept_dropout),
            nn.Linear(512, n_concepts),
        )

        # Style prediction head (concepts → styles)
        self.style_head = nn.Sequential(
            nn.Linear(n_concepts, 256),
            nn.ReLU(),
            nn.Dropout(style_dropout),
            nn.Linear(256, n_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights with proper strategies."""
        for module in [self.concept_head, self.style_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        concept_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CBM.

        Args:
            x: Input images [batch_size, 3, 224, 224]
            concept_labels: True concept labels for intervention [batch_size, n_concepts]

        Returns:
            concept_logits: Concept predictions [batch_size, n_concepts]
            style_logits: Style predictions [batch_size, n_classes]
        """
        # Extract features from backbone
        features = self.backbone(x)  # [batch_size, 1536]

        # Predict concepts
        concept_logits = self.concept_head(features)  # [batch_size, n_concepts]

        # For style prediction, use concepts (with optional intervention)
        if concept_labels is not None:
            # Use provided concept labels (intervention testing)
            concept_input = concept_labels.float()
        else:
            # Use predicted concepts (sigmoid for probability-like values)
            concept_input = torch.sigmoid(concept_logits)

        # Predict styles from concepts
        style_logits = self.style_head(concept_input)  # [batch_size, n_classes]

        return concept_logits, style_logits

    def get_concept_activations(
        self, x: torch.Tensor, threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Get interpretable concept activations for analysis.

        Args:
            x: Input images
            threshold: Threshold for concept activation

        Returns:
            Dictionary with concept analysis
        """
        with torch.no_grad():
            concept_logits, _ = self.forward(x)
            concept_probs = torch.sigmoid(concept_logits)

            # Active concepts per image
            active_concepts = (concept_probs > threshold).float()

            return {
                "concept_logits": concept_logits,
                "concept_probs": concept_probs,
                "active_concepts": active_concepts,
                "n_active_per_image": active_concepts.sum(dim=1),
                "concept_importance": concept_probs.mean(dim=0),
            }

    def freeze_concept_head(self):
        """Freeze concept head for style-only training."""
        for param in self.concept_head.parameters():
            param.requires_grad = False

    def unfreeze_concept_head(self):
        """Unfreeze concept head for joint training."""
        for param in self.concept_head.parameters():
            param.requires_grad = True

    def freeze_style_head(self):
        """Freeze style head for concept-only training."""
        for param in self.style_head.parameters():
            param.requires_grad = False

    def unfreeze_style_head(self):
        """Unfreeze style head for joint training."""
        for param in self.style_head.parameters():
            param.requires_grad = True

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration and parameter info."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "architecture": "CBM-EfficientNet-B3",
            "n_concepts": self.n_concepts,
            "n_classes": self.n_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "backbone_frozen": not any(
                p.requires_grad for p in self.backbone.parameters()
            ),
            "concept_head_frozen": not any(
                p.requires_grad for p in self.concept_head.parameters()
            ),
            "style_head_frozen": not any(
                p.requires_grad for p in self.style_head.parameters()
            ),
        }


def load_concept_list(concept_file: str = "model/cbm/data/final_concepts.json") -> list:
    """Load concept names from JSON file."""
    if os.path.exists(concept_file):
        with open(concept_file, "r") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif "selected_concepts" in data:
            return data["selected_concepts"]
        else:
            return list(data.keys())
    else:
        # Fallback to discriminative concepts
        fallback_file = "model/cbm/data/discriminative_concepts.json"
        if os.path.exists(fallback_file):
            with open(fallback_file, "r") as f:
                data = json.load(f)

            if isinstance(data, list):
                return data
            elif "selected_concepts" in data:
                return data["selected_concepts"]
            else:
                return list(data.keys())

    raise FileNotFoundError(
        f"No concept file found at {concept_file} or fallback location"
    )


def create_cbm_model(
    concept_file: str = "model/cbm/data/final_concepts.json", **model_kwargs
) -> ConceptBottleneckModel:
    """
    Factory function to create CBM model with auto-detected concept count.

    Args:
        concept_file: Path to concept list file
        **model_kwargs: Additional arguments for CBM model

    Returns:
        Initialized CBM model
    """
    concepts = load_concept_list(concept_file)
    n_concepts = len(concepts)

    model = ConceptBottleneckModel(n_concepts=n_concepts, **model_kwargs)

    print(f"✅ CBM created with {n_concepts} concepts")
    print(f"   Model info: {model.get_model_info()}")

    return model


if __name__ == "__main__":
    # Quick test of model architecture
    print("🧪 Testing CBM Model Architecture")
    print("=" * 40)

    # Create model
    model = create_cbm_model()

    # Test forward pass
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)

    print(f"\n📊 Testing forward pass...")
    print(f"   Input shape: {dummy_images.shape}")

    with torch.no_grad():
        concept_logits, style_logits = model(dummy_images)

    print(f"   Concept output: {concept_logits.shape}")
    print(f"   Style output: {style_logits.shape}")

    # Test concept analysis
    concept_info = model.get_concept_activations(dummy_images)
    print(f"   Active concepts per image: {concept_info['n_active_per_image']}")

    # Test staged training modes
    print(f"\n🎯 Testing staged training modes...")

    # Concept-only mode
    model.freeze_style_head()
    print(
        f"   Style head frozen: {not any(p.requires_grad for p in model.style_head.parameters())}"
    )

    # Style-only mode
    model.unfreeze_style_head()
    model.freeze_concept_head()
    print(
        f"   Concept head frozen: {not any(p.requires_grad for p in model.concept_head.parameters())}"
    )

    # Joint training mode
    model.unfreeze_concept_head()
    print(f"   Both heads active: {any(p.requires_grad for p in model.parameters())}")

    print(f"\n✅ Model architecture test complete!")
