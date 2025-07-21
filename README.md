# Art DNA

Machine learning project to classify art styles from paintings using machine learning.

## Prerequisites

- Python 3.10+
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

## Train & Evaluate

```bash
# Train model (first time only, ~10-15 minutes)
make train-model

# Evaluate model performance on test set
make evaluate-model
```

## API

- `POST /predict` - Upload image, returns art style predictions
- Live demo: [art-dna-api.run.app](https://art-dna-api-521843227251.europe-west1.run.app)

## Deployment

```bash
# Docker
make build_container_local
make run_container_local

# GCP Cloud Run
make build_for_production
make push_image_production
make deploy_to_cloud_run
```

## Commands

```bash
make help            # Show all commands
make build-dataset   # Build image dataset from CSV
make train-model     # Train and save VGG16 model
make evaluate-model  # Evaluate model on test set
make run-backend     # Run FastAPI backend
make run-frontend    # Run Streamlit frontend
```

## Team

- [Adriana]()
- [Anna](https://github.com/AnnaShe78)
- [Kristina](https://github.com/TinaKgn)
- [Marc](https://github.com/MarcRenard)
- [Michael](https://github.com/michaelsmueller)
