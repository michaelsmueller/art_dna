# Art DNA

Machine learning project to classify art styles from paintings using CNN transfer learning.

## Overview

- **Multi-Label Classification**: 18 art genres with independent probabilities (sigmoid vs softmax)
- **Explainable AI (CBM)**: Concept Bottleneck Model with 37 interpretable visual concepts
- **Visual Similarity Search**: DeiT embeddings find artworks with shared visual features
- **Interactive Frontend**: "Art Style Explorer" with real-time analysis
- **Production API**: Deployed on Google Cloud Run with 1,000+ artwork embeddings

## Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- Download [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time/data) dataset

## Setup

```bash
# Install dependencies
make install

# Extract dataset to raw_data/ directory
# Create dataset (first time only)
make build-dataset
```

## Run

```bash
# Run backend (terminal 1)
make run-backend

# Run frontend (terminal 2)
make run-frontend
```

Open `http://localhost:8501` to upload images and predict art styles.

## API Endpoints

**Production**: `https://art-dna-api-521843227251.europe-west1.run.app`

- `POST /predict` - Multi-label style classification (VGG16)
- `POST /predict_cbm` - Explainable classification with concepts (CBM)
- `GET /gradcam/{session_id}/style/{style}` - Style heatmap visualization
- `GET /gradcam/{session_id}/concept/{concept}` - Concept heatmap visualization
- `POST /similar` - Visual similarity search (top 5)
- `GET /describe` - Educational genre descriptions

## Deployment

```bash
# Docker
make build_container_local
make run_container_local

# GCP Cloud Run
make build_cloud
make deploy_to_cloud_run
```

## Commands

```bash
make help            # Show all commands
make build-dataset   # Build image dataset with 18 simplified genres
make train-model     # Train and save VGG16 model
make evaluate-model  # Evaluate model on test set
make run-backend     # Run FastAPI backend
make run-frontend    # Run Streamlit frontend
```

## Art Genres

Abstractionism, Art Nouveau, Baroque, Byzantine Art, Cubism, Expressionism, Impressionism, Mannerism, Muralism, Neoplasticism, Pop Art, Primitivism, Realism, Renaissance, Romanticism, Suprematism, Surrealism, Symbolism

## Architecture

- **Classification (VGG16)**: Fine-tuned VGG16 with weighted loss for class balance
- **Classification (CBM)**: EfficientNet-B3 → 37 visual concepts → 18 art styles
- **Explainability**: Grad-CAM visualizations for both styles and concepts
- **Similarity**: DeiT vision transformer (768-dim embeddings)
- **Dataset**: 7,600+ paintings, 50 artists, 18 simplified genres
- **Deployment**: Cloud Run + GCS model storage

## Team

- [Adriana](https://github.com/lady-hamster)
- [Anna](https://github.com/AnnaShe78)
- [Kristina](https://github.com/TinaKgn)
- [Marc](https://github.com/MarcRenard)
- [Michael](https://github.com/michaelsmueller)
