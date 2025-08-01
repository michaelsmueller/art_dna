# model/cbm_model.py
"""
Concept Bottleneck Model (CBM) for Art Style Classification
Uses EfficientNet-B3 backbone with concept bottleneck layer
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np


# Data augmentation removed - will be applied in training pipeline instead


def build_cbm_model(num_concepts=64, num_styles=18, input_shape=(224, 224, 3)):
    """
    Build Concept Bottleneck Model with EfficientNet-B3

    Architecture:
    Image → EfficientNet-B3 → GAP → Dense(concepts, sigmoid) → Dense(styles, sigmoid)

    Args:
        num_concepts: Number of concept neurons (e.g., 64)
        num_styles: Number of art style classes (18)
        input_shape: Input image shape

    Returns:
        Model with two outputs: [concepts, styles]
    """

    # EfficientNet-B3 backbone (frozen initially)
    backbone = EfficientNetB3(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    backbone.trainable = False  # Freeze for stable training

    # Model architecture
    inputs = layers.Input(shape=input_shape)

    # EfficientNet-B3 feature extraction (no augmentation)
    x = backbone(inputs)  # Output: (batch, H, W, 1536)

    # Global Average Pooling to flatten spatial dimensions
    x = layers.GlobalAveragePooling2D()(x)  # Output: (batch, 1536)

    # Concept bottleneck layer (sigmoid for multi-label concepts)
    concepts = layers.Dense(num_concepts, activation="sigmoid", name="concepts")(x)

    # Style classification from concepts (sigmoid for multi-label styles)
    styles = layers.Dense(num_styles, activation="sigmoid", name="styles")(concepts)

    # Create model with dual outputs
    model = Model(inputs=inputs, outputs=[concepts, styles], name="cbm_art_classifier")

    return model


def create_cbm_compiled(num_concepts=64, num_styles=18, learning_rate=1e-4):
    """
    Create and compile CBM model for training

    Returns:
        Compiled model ready for training
    """
    model = build_cbm_model(num_concepts=num_concepts, num_styles=num_styles)

    # Compile with binary crossentropy for both outputs
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"concepts": "binary_crossentropy", "styles": "binary_crossentropy"},
        metrics={"concepts": ["accuracy"], "styles": ["accuracy"]},
    )

    return model


def unfreeze_backbone_layers(model, num_layers=10):
    """
    Unfreeze last N layers of EfficientNet backbone for fine-tuning

    Args:
        model: CBM model
        num_layers: Number of layers to unfreeze from the end
    """
    backbone = None
    for layer in model.layers:
        if "efficientnetb3" in layer.name.lower():
            backbone = layer
            break

    if backbone is not None:
        # Unfreeze last num_layers layers
        for layer in backbone.layers[-num_layers:]:
            layer.trainable = True

        print(f"✅ Unfroze last {num_layers} layers of EfficientNet-B3")
    else:
        print("❌ EfficientNet-B3 backbone not found")


# Class names for reference
CLASS_NAMES = [
    "Abstractionism",
    "Art Nouveau",
    "Baroque",
    "Byzantine Art",
    "Cubism",
    "Expressionism",
    "Impressionism",
    "Mannerism",
    "Muralism",
    "Neoplasticism",
    "Pop Art",
    "Primitivism",
    "Realism",
    "Renaissance",
    "Romanticism",
    "Suprematism",
    "Surrealism",
    "Symbolism",
]


if __name__ == "__main__":
    # Test model creation
    print("🏗️ Testing CBM Model Creation")
    print("=" * 35)

    model = build_cbm_model(num_concepts=64, num_styles=18)

    print(f"✅ Model created successfully")
    print(f"📊 Input shape: {model.input_shape}")
    print(f"📊 Output shapes: {[output.shape for output in model.outputs]}")
    print(f"📊 Total parameters: {model.count_params():,}")

    # Test prediction
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    concepts_pred, styles_pred = model.predict(dummy_input, verbose=0)

    print(f"\n🧪 Test Prediction:")
    print(f"   Concepts shape: {concepts_pred.shape}")
    print(f"   Styles shape: {styles_pred.shape}")
    print(f"   Concepts range: [{concepts_pred.min():.3f}, {concepts_pred.max():.3f}]")
    print(f"   Styles range: [{styles_pred.min():.3f}, {styles_pred.max():.3f}]")

    print(f"\n🎯 Ready for training!")
