# model/cbm/train_concept_cbm.py
"""
CBM training with concept bottleneck layer
Dual outputs: concepts + styles
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def process_image(file_path, labels):
    """Simple image processing"""
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, labels


def create_dataset(df, class_names, batch_size=16, shuffle=True):
    """Create dataset with dual outputs (concepts + styles)"""
    image_paths = df["image_path"].values
    style_labels = df[class_names].values.astype(np.float32)

    # Create dummy concept labels for now (we'll improve this later)
    num_concepts = 64
    concept_labels = np.random.rand(len(df), num_concepts).astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, (concept_labels, style_labels))
    )

    def process_with_dual_labels(file_path, labels):
        concept_labels, style_labels = labels
        image, _ = process_image(file_path, None)
        return image, {"concepts": concept_labels, "styles": style_labels}

    dataset = dataset.map(process_with_dual_labels, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def train_concept_cbm():
    """Train CBM with concept bottleneck"""
    print("🚀 CBM Training with Concept Bottleneck")
    print("=" * 40)

    # Load data
    df = pd.read_csv("raw_data/final_df.csv")
    print(f"📂 Dataset: {len(df)} images")

    # Import CBM model
    from cbm_model import create_cbm_compiled, CLASS_NAMES

    # Check data
    print(f"📊 Sample class distribution:")
    for genre in CLASS_NAMES[:5]:
        count = df[genre].sum()
        print(f"   {genre}: {count} images")
    print("   ...")

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

    print(f"📊 Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Create datasets with dual outputs
    batch_size = 16
    train_dataset = create_dataset(train_df, CLASS_NAMES, batch_size, shuffle=True)
    val_dataset = create_dataset(val_df, CLASS_NAMES, batch_size, shuffle=False)

    # Create CBM model with dual outputs
    model = create_cbm_compiled(num_concepts=64, num_styles=18, learning_rate=1e-4)

    print(f"🏗️ CBM Model created:")
    print(f"   Input: {model.input_shape}")
    print(f"   Outputs: {[output.shape for output in model.outputs]}")
    print(f"   Parameters: {model.count_params():,}")

    # Stage 1: Train styles only (ignore concepts)
    print(f"\n🎯 Stage 1: Training Styles (ignoring concepts)")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={"concepts": "binary_crossentropy", "styles": "binary_crossentropy"},
        loss_weights={
            "concepts": 0.0,  # Ignore concepts for now
            "styles": 1.0,  # Focus on styles
        },
        metrics={"concepts": ["accuracy"], "styles": ["accuracy"]},
    )

    # Simple callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "model/cbm/concept_cbm_stage1.keras",  # Save in CBM folder
            monitor="val_styles_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_styles_loss",
            patience=5,
            mode="min",
            verbose=1,
            restore_best_weights=True,
        ),
    ]

    # Train Stage 1
    print(f"\n🏃‍♂️ Training Stage 1 (5 epochs)...")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=5,
        callbacks=callbacks,
        verbose=1,
    )

    # Results
    print(f"\n🎉 Stage 1 completed!")
    final_acc = history.history["val_styles_accuracy"][-1]
    final_loss = history.history["val_styles_loss"][-1]
    print(f"   Final validation accuracy: {final_acc:.4f}")
    print(f"   Final validation loss: {final_loss:.4f}")

    # Test predictions
    print(f"\n🔍 Testing dual outputs...")
    sample_batch = next(iter(val_dataset))
    sample_images, sample_labels = sample_batch

    concept_preds, style_preds = model.predict(sample_images[:3], verbose=0)

    print(f"Concept predictions shape: {concept_preds.shape}")
    print(f"Style predictions shape: {style_preds.shape}")
    print(f"Style prediction range: [{style_preds.min():.3f}, {style_preds.max():.3f}]")

    # Check saturation
    saturated = (style_preds > 0.99).sum() + (style_preds < 0.01).sum()
    total = style_preds.size
    saturation_rate = saturated / total

    if saturation_rate < 0.5:
        print(f"✅ Good predictions: {saturation_rate:.1%} saturation")
    else:
        print(f"⚠️ High saturation: {saturation_rate:.1%}")

    return model, history


if __name__ == "__main__":
    model, history = train_concept_cbm()
    print("\n✅ Concept CBM Stage 1 completed!")
    print("🔄 Next: Train concept layer with real concept labels")
