#!/usr/bin/env python3
"""
Package and prepare CBM v1 for deployment.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime


def create_deployment_package(version="1.0"):
    """Create deployment package for CBM model."""

    print(f"📦 Creating deployment package for CBM v{version}...")

    # Create release directory
    release_dir = Path(f"releases/cbm-v{version}")
    release_dir.mkdir(parents=True, exist_ok=True)

    # Files to include
    files_to_copy = {
        "model/cbm/models/cbm_weighted_best.pth": "cbm_model.pth",
        "model/cbm/val_thresholds.npy": "thresholds.npy",
        "model/cbm/data/final_concepts.json": "concepts.json",
    }

    # Copy files
    for src, dst in files_to_copy.items():
        if os.path.exists(src):
            shutil.copy2(src, release_dir / dst)
            print(f"✅ Copied {src} → {dst}")
        else:
            print(f"⚠️  Missing {src}")

    # Create metadata
    metadata = {
        "version": version,
        "model_architecture": "cbm-efficientnet-b3",
        "n_concepts": 37,
        "n_classes": 18,
        "test_metrics": {"style_f1_weighted": 0.601, "concept_f1": 0.664},
        "training_info": {
            "backbone": "efficientnet-b3",
            "epochs": 15,
            "concept_weight": 0.3,
            "weight_cap": 5.0,
        },
        "created_at": datetime.utcnow().isoformat() + "Z",
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
    }

    with open(release_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Create README
    readme_content = f"""# CBM v{version} Deployment Package

## Contents
- `cbm_model.pth`: Trained model weights
- `thresholds.npy`: Optimal classification thresholds
- `concepts.json`: Concept definitions and names
- `metadata.json`: Model metadata and performance metrics

## Performance
- Style F1 (weighted): 0.601
- Concept F1: 0.664
- Exceeds VGG16 baseline by 39%

## Usage
```python
import torch
import numpy as np
import json

# Load model
from model.cbm_model import ConceptBottleneckModel

model = ConceptBottleneckModel(n_concepts=37, n_classes=18)
checkpoint = torch.load('cbm_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load thresholds
thresholds = np.load('thresholds.npy')

# Load concepts
with open('concepts.json', 'r') as f:
    concepts = json.load(f)['selected_concepts']
```

## API Integration
This model is integrated into the FastAPI backend at `/predict_cbm` endpoint.
"""

    with open(release_dir / "README.md", "w") as f:
        f.write(readme_content)

    # Create deployment config for Cloud Run
    config = {
        "model_path": "cbm_model.pth",
        "thresholds_path": "thresholds.npy",
        "concepts_path": "concepts.json",
        "preprocessing": {
            "image_size": [224, 224],
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    }

    with open(release_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create tarball
    print("\n📦 Creating tarball...")
    os.system(f"cd releases && tar -czf cbm-v{version}.tar.gz cbm-v{version}/")

    print(f"\n✅ Deployment package created: releases/cbm-v{version}.tar.gz")

    # Print upload commands
    print("\n📤 To upload to GCS:")
    print(
        f"gsutil cp releases/cbm-v{version}.tar.gz gs://YOUR_BUCKET/models/cbm/v{version}/"
    )

    print("\n📊 To log in MLflow:")
    print(
        f"""
import mlflow
mlflow.set_tracking_uri("YOUR_MLFLOW_URI")
mlflow.set_experiment("cbm-production")

with mlflow.start_run(run_name=f"cbm-v{version}"):
    mlflow.log_artifact("releases/cbm-v{version}/")
    mlflow.log_metrics({metadata['test_metrics']})
    mlflow.set_tag("version", "{version}")
"""
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="1.0", help="Version number")
    args = parser.parse_args()

    create_deployment_package(args.version)
