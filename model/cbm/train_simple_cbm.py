# model/train_simple_cbm.py
"""
Simple CBM training - start basic and build up
Single output (styles only), no concept bottleneck yet
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os


def process_image(file_path, labels):
    """Simple image processing"""
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, labels


def create_dataset(df, class_names, batch_size=16, shuffle=True):
    """Create simple dataset"""
    image_paths = df["image_path"].values
    style_labels = df[class_names].values.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, style_labels))
    dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_simple_model():
    """Create simple EfficientNet model - styles only"""
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras import layers, Model

    # EfficientNet backbone
    backbone = EfficientNetB3(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    backbone.trainable = False  # Frozen

    # Simple model
    inputs = layers.Input(shape=(224, 224, 3))
    x = backbone(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(18, activation="sigmoid", name="styles")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def train_simple():
    """Simple training loop"""
    print("🚀 Simple CBM Training - Styles Only")
    print("=" * 35)

    # Load data
    df = pd.read_csv("raw_data/final_df.csv")
    print(f"📂 Dataset: {len(df)} images")

    # Class names
    class_names = [
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

    # Check data
    print(f"📊 Class distribution:")
    for genre in class_names[:5]:  # Show first 5
        count = df[genre].sum()
        print(f"   {genre}: {count} images")
    print("   ...")

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

    print(f"📊 Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Create datasets
    batch_size = 16
    train_dataset = create_dataset(train_df, class_names, batch_size, shuffle=True)
    val_dataset = create_dataset(val_df, class_names, batch_size, shuffle=False)

    # Create model
    model = create_simple_model()

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print(f"🏗️ Model created:")
    print(f"   Input: {model.input_shape}")
    print(f"   Output: {model.output_shape}")
    print(f"   Parameters: {model.count_params():,}")

    # Simple callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "model/simple_cbm.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
            verbose=1,
            restore_best_weights=True,
        ),
    ]

    # Train
    print(f"\n🏃‍♂️ Training for 5 epochs...")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=5,
        callbacks=callbacks,
        verbose=1,
    )

    # Results
    print(f"\n🎉 Training completed!")
    final_acc = history.history["val_accuracy"][-1]
    final_loss = history.history["val_loss"][-1]
    print(f"   Final validation accuracy: {final_acc:.4f}")
    print(f"   Final validation loss: {final_loss:.4f}")

    # Test predictions
    print(f"\n🔍 Testing predictions...")
    sample_batch = next(iter(val_dataset))
    sample_images, sample_labels = sample_batch
    predictions = model.predict(sample_images[:3], verbose=0)

    print(f"Prediction shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")

    # Check saturation
    saturated = (predictions > 0.99).sum() + (predictions < 0.01).sum()
    total = predictions.size
    saturation_rate = saturated / total

    if saturation_rate < 0.5:
        print(f"✅ Good predictions: {saturation_rate:.1%} saturation")
    else:
        print(f"⚠️ High saturation: {saturation_rate:.1%}")

    return model, history


if __name__ == "__main__":
    model, history = train_simple()
    print("\n✅ Simple training completed!")
    print("🔄 Next: Add concept bottleneck layer")
