from data_loader import get_data_generators
from model_builder import build_model
import os
import sys

model_path = "model/art_style_classifier.keras"

if os.path.exists(model_path):
    print(f"❌ Model already exists at: {model_path}")
    print("ℹ️ To retrain, delete the model manually or rename it.")
    sys.exit(1)

csv_path = "raw_data/labeled_data.csv"

train_gen, val_gen = get_data_generators(csv_path)

model = build_model(num_classes=len(train_gen.class_indices))

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

model.class_names = list(train_gen.class_indices.keys())
model.save("model/art_style_classifier.keras")
print("✅ Model saved to model/art_style_classifier.keras")
