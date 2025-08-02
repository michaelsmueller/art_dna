"""
Test concept-specific Grad-CAM using pilot extraction data.
Trains mini concept classifiers and visualizes what activates each concept.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# Load pilot data
def load_pilot_data():
    """Load pilot concept extraction results."""
    with open("model/cbm/pilot_results.json", "r") as f:
        pilot_data = json.load(f)

    # Combine both runs for more data
    all_results = {}
    for img_path, scores in pilot_data["run1"].items():
        all_results[f"{img_path}_run1"] = scores
    for img_path, scores in pilot_data["run2"].items():
        all_results[f"{img_path}_run2"] = scores

    return all_results, pilot_data["concepts"]


# Select concepts with good variation for testing
def select_test_concepts(concept_scores, concepts, n_concepts=6):
    """Select concepts with good score variation (not too rare/common)."""
    concept_stats = {}

    for concept in concepts:
        scores = [result.get(concept, 0) for result in concept_scores.values()]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        presence_rate = sum(1 for s in scores if s >= 0.7) / len(scores)

        # Good concepts have:
        # - Not too rare (>5% presence) or too common (<95% presence)
        # - Good variation (std > 0.2)
        if 0.05 < presence_rate < 0.95 and std_score > 0.2:
            concept_stats[concept] = {
                "mean": mean_score,
                "std": std_score,
                "presence": presence_rate,
            }

    # Sort by standard deviation (most variation)
    sorted_concepts = sorted(
        concept_stats.items(), key=lambda x: x[1]["std"], reverse=True
    )
    selected = [concept for concept, stats in sorted_concepts[:n_concepts]]

    print(f"🎯 Selected {len(selected)} concepts for Grad-CAM testing:")
    for concept in selected:
        stats = concept_stats[concept]
        print(
            f"  {concept}: μ={stats['mean']:.2f}, σ={stats['std']:.2f}, presence={stats['presence']:.1%}"
        )

    return selected


class ConceptDataset(Dataset):
    """Dataset for concept classification."""

    def __init__(self, image_paths, concept_scores, target_concept, transform=None):
        # Extract paths without run suffix
        self.image_paths = []
        self.labels = []

        for path_with_run, scores in concept_scores.items():
            # Remove _run1/_run2 suffix to get actual path
            actual_path = path_with_run.replace("_run1", "").replace("_run2", "")

            if os.path.exists(actual_path):
                self.image_paths.append(actual_path)
                # Convert concept score to binary label (>= 0.7 = positive)
                score = scores.get(target_concept, 0)
                self.labels.append(1 if score >= 0.7 else 0)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], self.image_paths[idx]


class SimpleConceptClassifier(nn.Module):
    """Simple binary concept classifier."""

    def __init__(self):
        super().__init__()
        # Use small EfficientNet for speed
        self.backbone = efficientnet_b0(pretrained=True)
        # Replace classifier head for binary classification
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(1280, 1)  # Binary output
        )

    def forward(self, x):
        return self.backbone(x)


def train_concept_classifier(concept, concept_scores, max_epochs=20):
    """Train a simple classifier for one concept."""
    print(f"\n🔄 Training classifier for: {concept}")

    # Data transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Dataset and loader
    dataset = ConceptDataset([], concept_scores, concept, transform)
    if len(dataset) < 5:
        print(f"  ⚠️  Too few samples ({len(dataset)}) for {concept}")
        return None

    # Check class balance
    labels = [dataset[i][1] for i in range(len(dataset))]
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count

    print(f"  Samples: {len(dataset)} (pos: {pos_count}, neg: {neg_count})")

    if pos_count == 0 or neg_count == 0:
        print(f"  ⚠️  No class variation for {concept}")
        return None

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Model, loss, optimizer
    model = SimpleConceptClassifier()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(max_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        if epoch % 5 == 0:  # Print every 5 epochs
            print(f"    Epoch {epoch:2d}: Loss={epoch_loss:.3f}, Acc={accuracy:.3f}")

    print(f"  ✅ Final accuracy: {accuracy:.3f}")
    return model


def test_concept_gradcam(model, concept, concept_scores, n_samples=3):
    """Test Grad-CAM on concept classifier."""
    print(f"\n🔍 Testing Grad-CAM for: {concept}")

    # Setup Grad-CAM
    target_layers = [model.backbone.features[-1]]  # Last conv layer
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    # Get some test images
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = ConceptDataset([], concept_scores, concept, transform)

    # Test on both positive and negative examples
    pos_indices = [i for i in range(len(dataset)) if dataset[i][1] == 1]
    neg_indices = [i for i in range(len(dataset)) if dataset[i][1] == 0]

    test_indices = pos_indices[: n_samples // 2] + neg_indices[: n_samples // 2]
    test_indices = test_indices[:n_samples]  # Ensure we don't exceed n_samples

    print(f"  Testing on {len(test_indices)} samples")

    for i, idx in enumerate(test_indices):
        image_tensor, label, image_path = dataset[idx]

        # Generate Grad-CAM
        input_tensor = image_tensor.unsqueeze(0)
        targets = [ClassifierOutputTarget(0)]  # Target the single output

        # Get the grayscale CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Remove batch dimension

        # Load original image for visualization
        original_image = Image.open(image_path).convert("RGB")
        original_image = original_image.resize((224, 224))
        original_np = np.array(original_image) / 255.0

        # Create visualization
        visualization = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)

        # Print results
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()

        label_str = "POSITIVE" if label == 1 else "NEGATIVE"
        prediction_str = f"pred={prob:.3f}"

        print(f"    Sample {i+1}: {label_str} example, {prediction_str}")
        print(f"      Path: {os.path.basename(image_path)}")

        # Save visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.imshow(original_np)
        ax1.set_title(f"Original\n{label_str}")
        ax1.axis("off")

        ax2.imshow(visualization)
        ax2.set_title(f"Grad-CAM: {concept}\nPred: {prob:.3f}")
        ax2.axis("off")

        output_path = f"model/cbm/gradcam_{concept}_{i+1}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"      Saved: {output_path}")


def main():
    print("🧪 CONCEPT GRAD-CAM TEST")
    print("=" * 50)

    # Load pilot data
    concept_scores, all_concepts = load_pilot_data()
    print(f"Loaded {len(concept_scores)} samples with {len(all_concepts)} concepts")

    # Select concepts to test
    test_concepts = select_test_concepts(concept_scores, all_concepts, n_concepts=5)

    if not test_concepts:
        print("❌ No suitable concepts found for testing")
        return

    # Test each concept
    trained_models = {}
    for concept in test_concepts:
        model = train_concept_classifier(concept, concept_scores, max_epochs=15)
        if model is not None:
            trained_models[concept] = model

    # Generate Grad-CAM visualizations
    print(f"\n🎨 GENERATING GRAD-CAM VISUALIZATIONS")
    print("=" * 50)

    for concept, model in trained_models.items():
        test_concept_gradcam(model, concept, concept_scores, n_samples=4)

    print(f"\n✅ Concept Grad-CAM test complete!")
    print(f"📁 Visualizations saved to: model/cbm/gradcam_*.png")
    print(f"🎯 Successfully tested {len(trained_models)} concepts")


if __name__ == "__main__":
    main()
