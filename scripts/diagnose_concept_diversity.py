#!/usr/bin/env python3
"""
Comprehensive diagnosis of concept diversity and activation patterns in the CBM model.
"""

import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import random

# Import our modules
from model.cbm_model import ConceptBottleneckModel
from model.cbm.concept_dataset import get_concept_data_loaders


class ConceptDiversityAnalyzer:
    def __init__(self, model_path, device="cpu"):
        """Initialize analyzer with trained model."""
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()

        # Load concept names
        with open("model/cbm/data/final_concepts.json", "r") as f:
            self.concept_names = json.load(f)["selected_concepts"]

        # Get data loaders
        loaders = get_concept_data_loaders(batch_size=32, num_workers=0)
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.test_loader = loaders["test"]

    def _load_model(self, model_path):
        """Load the trained CBM model."""
        print(f"📦 Loading model from {model_path}")
        model = ConceptBottleneckModel(n_concepts=37, n_classes=18)
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        return model

    def extract_concept_activations(self, loader, n_samples=500):
        """Extract concept activations from a data loader."""
        print(f"🔄 Extracting concept activations from {n_samples} samples...")

        all_concepts = []
        all_styles = []
        all_true_styles = []

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if len(all_concepts) >= n_samples:
                    break

                # Unpack batch
                images, style_labels, concept_labels = batch
                images = images.to(self.device)

                # Forward pass
                concept_logits, style_logits = self.model(images)
                concept_probs = torch.sigmoid(concept_logits)
                style_probs = torch.sigmoid(style_logits)

                all_concepts.append(concept_probs.cpu().numpy())
                all_styles.append(style_probs.cpu().numpy())
                all_true_styles.append(style_labels.cpu().numpy())

        # Concatenate all batches
        all_concepts = np.concatenate(all_concepts, axis=0)[:n_samples]
        all_styles = np.concatenate(all_styles, axis=0)[:n_samples]
        all_true_styles = np.concatenate(all_true_styles, axis=0)[:n_samples]

        print(f"✅ Extracted {all_concepts.shape[0]} samples")
        return all_concepts, all_styles, all_true_styles

    def analyze_concept_correlations(self, concept_activations):
        """Analyze pairwise correlations between concepts."""
        print("\n📊 Analyzing concept correlations...")

        # Compute correlation matrix
        corr_matrix = np.corrcoef(concept_activations.T)

        # Find highly correlated pairs
        n_concepts = len(self.concept_names)
        high_corr_pairs = []

        for i in range(n_concepts):
            for j in range(i + 1, n_concepts):
                corr = corr_matrix[i, j]
                if abs(corr) > 0.8:
                    high_corr_pairs.append(
                        (self.concept_names[i], self.concept_names[j], corr)
                    )

        # Plot correlation matrix
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            annot=False,  # Too many to annotate
            xticklabels=self.concept_names,
            yticklabels=self.concept_names,
            cmap="RdBu_r",
            center=0,
            mask=mask,
            square=True,
            cbar_kws={"shrink": 0.8},
        )

        plt.title("Concept Activation Correlations (Full Dataset)", fontsize=16)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig("concept_diversity_analysis/concept_correlations_full.png", dpi=150)
        plt.close()

        return corr_matrix, high_corr_pairs

    def compute_concept_diversity_metrics(self, concept_activations):
        """Compute various diversity metrics."""
        print("\n📏 Computing diversity metrics...")

        metrics = {}

        # 1. Average pairwise correlation
        corr_matrix = np.corrcoef(concept_activations.T)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_correlation = np.abs(corr_matrix[mask]).mean()
        metrics["avg_absolute_correlation"] = float(avg_correlation)

        # 2. Effective rank (eigenvalue-based)
        cov_matrix = np.cov(concept_activations.T)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros
        normalized_eigenvalues = eigenvalues / eigenvalues.sum()
        effective_rank = np.exp(entropy(normalized_eigenvalues))
        metrics["effective_rank"] = float(effective_rank)
        metrics["max_possible_rank"] = len(self.concept_names)

        # 3. Activation sparsity
        mean_activation = concept_activations.mean(axis=0)
        sparsity = (mean_activation < 0.1).sum() / len(mean_activation)
        metrics["concept_sparsity"] = float(sparsity)

        # 4. Redundancy score (% of concepts with correlation > 0.9)
        high_corr_count = (np.abs(corr_matrix) > 0.9).sum() - len(self.concept_names)
        total_pairs = len(self.concept_names) * (len(self.concept_names) - 1) / 2
        redundancy = high_corr_count / (2 * total_pairs)
        metrics["redundancy_score"] = float(redundancy)

        return metrics

    def visualize_concept_space(self, concept_activations, method="tsne"):
        """Visualize concept activations in 2D."""
        print(f"\n🎨 Visualizing concept space with {method.upper()}...")

        # Transpose to have concepts as points
        concept_patterns = concept_activations.T  # shape: (n_concepts, n_samples)

        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=10)
        else:  # PCA
            reducer = PCA(n_components=2)

        concept_2d = reducer.fit_transform(concept_patterns)

        # Create scatter plot
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            concept_2d[:, 0],
            concept_2d[:, 1],
            c=range(len(self.concept_names)),
            cmap="tab20",
            s=100,
            alpha=0.7,
        )

        # Add labels
        for i, name in enumerate(self.concept_names):
            plt.annotate(
                name,
                (concept_2d[i, 0], concept_2d[i, 1]),
                fontsize=8,
                alpha=0.8,
                xytext=(5, 5),
                textcoords="offset points",
            )

        plt.title(f"Concept Space Visualization ({method.upper()})", fontsize=16)
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"concept_diversity_analysis/concept_space_{method}.png", dpi=150)
        plt.close()

        return concept_2d

    def analyze_concept_clustering(self, concept_activations):
        """Identify clusters of similar concepts."""
        print("\n🎯 Analyzing concept clusters...")

        # Compute distance matrix
        concept_patterns = concept_activations.T
        distances = pdist(concept_patterns, metric="correlation")
        distance_matrix = squareform(distances)

        # Hierarchical clustering
        from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

        linkage_matrix = linkage(distances, method="average")

        # Plot dendrogram
        plt.figure(figsize=(16, 8))
        dendrogram(
            linkage_matrix,
            labels=self.concept_names,
            leaf_rotation=90,
            leaf_font_size=10,
        )
        plt.title("Concept Clustering Dendrogram", fontsize=16)
        plt.xlabel("Concept")
        plt.ylabel("Correlation Distance")
        plt.tight_layout()
        plt.savefig("concept_diversity_analysis/concept_dendrogram.png", dpi=150)
        plt.close()

        # Get clusters at different thresholds
        clusters_strict = fcluster(linkage_matrix, 0.1, criterion="distance")
        clusters_moderate = fcluster(linkage_matrix, 0.3, criterion="distance")

        return clusters_strict, clusters_moderate

    def sample_diverse_predictions(self, n_samples=10):
        """Sample and visualize predictions with concept diversity info."""
        print(f"\n🖼️ Sampling {n_samples} predictions...")

        # Get random samples from test set
        test_iter = iter(self.test_loader)
        batch = next(test_iter)
        images, style_labels, concept_labels = batch

        # Select random samples
        indices = random.sample(range(len(images)), min(n_samples, len(images)))

        diversity_info = []

        with torch.no_grad():
            for idx in indices:
                img = images[idx : idx + 1].to(self.device)

                # Get predictions
                concept_logits, style_logits = self.model(img)
                concept_probs = torch.sigmoid(concept_logits).cpu().numpy()[0]

                # Find top activated concepts
                top_indices = np.argsort(concept_probs)[-5:][::-1]
                top_concepts = [
                    (self.concept_names[i], concept_probs[i]) for i in top_indices
                ]

                # Compute diversity of top concepts
                top_concept_patterns = concept_probs[top_indices]
                pairwise_diff = np.abs(
                    np.subtract.outer(top_concept_patterns, top_concept_patterns)
                )
                avg_difference = pairwise_diff[
                    np.triu_indices_from(pairwise_diff, k=1)
                ].mean()

                diversity_info.append(
                    {
                        "top_concepts": top_concepts,
                        "avg_difference": avg_difference,
                        "activation_entropy": entropy(concept_probs + 1e-10),
                    }
                )

        return diversity_info

    def generate_report(self, output_dir="concept_diversity_analysis"):
        """Generate comprehensive diversity analysis report."""
        Path(output_dir).mkdir(exist_ok=True)

        # Extract activations from all splits
        train_concepts, _, _ = self.extract_concept_activations(
            self.train_loader, n_samples=1000
        )
        val_concepts, _, _ = self.extract_concept_activations(
            self.val_loader, n_samples=500
        )
        test_concepts, _, _ = self.extract_concept_activations(
            self.test_loader, n_samples=500
        )

        # Analyze correlations
        corr_matrix, high_corr_pairs = self.analyze_concept_correlations(train_concepts)

        # Compute diversity metrics
        metrics = {
            "train": self.compute_concept_diversity_metrics(train_concepts),
            "val": self.compute_concept_diversity_metrics(val_concepts),
            "test": self.compute_concept_diversity_metrics(test_concepts),
        }

        # Visualize concept space
        self.visualize_concept_space(train_concepts, method="tsne")
        self.visualize_concept_space(train_concepts, method="pca")

        # Analyze clustering
        clusters_strict, clusters_moderate = self.analyze_concept_clustering(
            train_concepts
        )

        # Sample predictions
        sample_diversity = self.sample_diverse_predictions()

        # Generate text report
        report = f"""
# Concept Diversity Analysis Report

## Summary Metrics

### Training Set
- Average Absolute Correlation: {metrics['train']['avg_absolute_correlation']:.3f}
- Effective Rank: {metrics['train']['effective_rank']:.1f} / {metrics['train']['max_possible_rank']}
- Concept Sparsity: {metrics['train']['concept_sparsity']:.2%}
- Redundancy Score: {metrics['train']['redundancy_score']:.2%}

### Validation Set
- Average Absolute Correlation: {metrics['val']['avg_absolute_correlation']:.3f}
- Effective Rank: {metrics['val']['effective_rank']:.1f} / {metrics['val']['max_possible_rank']}

### Test Set
- Average Absolute Correlation: {metrics['test']['avg_absolute_correlation']:.3f}
- Effective Rank: {metrics['test']['effective_rank']:.1f} / {metrics['test']['max_possible_rank']}

## Highly Correlated Concept Pairs (|r| > 0.8)
Found {len(high_corr_pairs)} pairs:
"""
        for c1, c2, corr in sorted(
            high_corr_pairs, key=lambda x: abs(x[2]), reverse=True
        )[:20]:
            report += f"- {c1} ↔ {c2}: r = {corr:.3f}\n"

        # Cluster analysis
        n_clusters_strict = len(np.unique(clusters_strict))
        n_clusters_moderate = len(np.unique(clusters_moderate))

        report += f"""
## Concept Clustering
- Strict clustering (d < 0.1): {n_clusters_strict} clusters
- Moderate clustering (d < 0.3): {n_clusters_moderate} clusters

## Recommendations
"""

        if metrics["train"]["avg_absolute_correlation"] > 0.6:
            report += (
                "⚠️ HIGH CORRELATION: Many concepts are learning similar features\n"
            )

        if metrics["train"]["effective_rank"] < 20:
            report += "⚠️ LOW EFFECTIVE RANK: Concept space has low dimensionality\n"

        if metrics["train"]["redundancy_score"] > 0.3:
            report += "⚠️ HIGH REDUNDANCY: Many concept pairs are nearly identical\n"

        # Save report
        with open(f"{output_dir}/diversity_report.txt", "w") as f:
            f.write(report)

        # Save metrics as JSON
        with open(f"{output_dir}/diversity_metrics.json", "w") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "n_high_corr_pairs": len(high_corr_pairs),
                    "n_clusters_strict": int(n_clusters_strict),
                    "n_clusters_moderate": int(n_clusters_moderate),
                },
                f,
                indent=2,
            )

        print(f"\n✅ Report saved to {output_dir}/")
        print("\n" + "=" * 60)
        print(report)

        return metrics, high_corr_pairs


if __name__ == "__main__":
    # Run comprehensive analysis
    analyzer = ConceptDiversityAnalyzer(
        model_path="model/cbm/models/cbm_weighted_best.pth",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("🚀 Starting comprehensive concept diversity analysis...")
    analyzer.generate_report()

    print("\n📊 View generated plots:")
    print("- concept_diversity_analysis/concept_correlations_full.png")
    print("- concept_diversity_analysis/concept_space_tsne.png")
    print("- concept_diversity_analysis/concept_space_pca.png")
    print("- concept_diversity_analysis/concept_dendrogram.png")
