#!/usr/bin/env python3
"""
Vertex AI Training Job Launcher for CBM

Usage:
    python scripts/launch_vertex.py --gpu-type T4 --preemptible
    python scripts/launch_vertex.py --gpu-type A100 --epochs 20
    python scripts/launch_vertex.py --dry-run  # Just show the config
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_project_config():
    """Get GCP project configuration from environment"""
    project = os.getenv("GCP_PROJECT")
    region = os.getenv("VERTEX_AI_REGION", "europe-west2")

    if not project:
        print("❌ GCP_PROJECT environment variable not set")
        sys.exit(1)

    return project, region


def generate_job_config(args):
    """Generate Vertex AI custom job configuration"""
    project, region = get_project_config()

    # Generate unique job name
    timestamp = datetime.now().strftime("%m%d-%H%M")
    job_name = f"cbm-{args.gpu_type.lower()}-{timestamp}"

    # GPU configurations
    gpu_configs = {
        "T4": {
            "machine_type": "n1-standard-4",
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": 1,
        },
        "A100": {
            "machine_type": "a2-highgpu-1g",
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 1,
        },
    }

    gpu_config = gpu_configs[args.gpu_type]

    # Container image
    image_uri = f"gcr.io/{project}/art-dna-vertex:latest"

    # Training arguments
    training_args = [
        "python",
        args.script,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--backbone-lr",
        str(args.lr_backbone),
        "--head-lr",
        str(args.lr_heads),
        "--output-dir",
        "gs://art-dna-ml-models/cbm/vertex-runs",
    ]

    if args.resume_from:
        training_args.extend(["--checkpoint", args.resume_from])

    # Job configuration (only jobSpec for config file)
    job_config = {
        "workerPoolSpecs": [
            {
                "machineSpec": {
                    "machineType": gpu_config["machine_type"],
                    "acceleratorType": gpu_config["accelerator_type"],
                    "acceleratorCount": gpu_config["accelerator_count"],
                },
                "replicaCount": 1,
                "containerSpec": {
                    "imageUri": image_uri,
                    "args": training_args,
                    "env": [
                        {
                            "name": "MLFLOW_TRACKING_URI",
                            "value": os.getenv("MLFLOW_TRACKING_URI"),
                        },
                        {
                            "name": "MLFLOW_TRACKING_USERNAME",
                            "value": os.getenv("MLFLOW_TRACKING_USERNAME"),
                        },
                        {
                            "name": "MLFLOW_TRACKING_PASSWORD",
                            "value": os.getenv("MLFLOW_TRACKING_PASSWORD"),
                        },
                        {"name": "VERTEX_AI_TRAINING", "value": "true"},
                        {"name": "GCP_PROJECT", "value": project},
                        {"name": "GCP_REGION", "value": region},
                    ],
                },
            }
        ]
    }

    # Add spot (preemptible) scheduling if requested
    if args.preemptible:
        job_config["scheduling"] = {
            "restartJobOnWorkerRestart": True,
            "timeout": "43200s",  # 12 hours max
        }

    return job_name, job_config


def submit_job(job_name, job_config, dry_run=False):
    """Submit job to Vertex AI"""
    project, region = get_project_config()

    if dry_run:
        print(f"🔍 Dry run - Job configuration for '{job_name}':")
        print(json.dumps(job_config, indent=2))
        return

    # Write job config to temporary file
    config_file = f"/tmp/{job_name}_config.json"
    with open(config_file, "w") as f:
        json.dump(job_config, f, indent=2)

    try:
        print(f"🚀 Submitting job '{job_name}' to Vertex AI...")

        # Submit job using gcloud
        cmd = [
            "gcloud",
            "ai",
            "custom-jobs",
            "create",
            "--display-name",
            job_name,
            "--region",
            region,
            "--config",
            config_file,
            "--project",
            project,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Job '{job_name}' submitted successfully!")
            print(
                f"📊 Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project}"
            )
            print(
                f"📋 View logs with: gcloud ai custom-jobs describe {job_name} --region={region}"
            )
        else:
            print(f"❌ Failed to submit job: {result.stderr}")

    except Exception as e:
        print(f"❌ Error submitting job: {e}")
    finally:
        # Clean up temp file
        if os.path.exists(config_file):
            os.remove(config_file)


def main():
    parser = argparse.ArgumentParser(description="Launch CBM training on Vertex AI")

    # Job configuration
    parser.add_argument(
        "--gpu-type",
        choices=["T4", "A100"],
        default="T4",
        help="GPU type for training (default: T4)",
    )
    parser.add_argument(
        "--preemptible",
        action="store_true",
        help="Use preemptible instances (cheaper but can be interrupted)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=12, help="Number of training epochs (default: 12)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=1e-5,
        help="Learning rate for backbone (default: 1e-5)",
    )
    parser.add_argument(
        "--lr-heads",
        type=float,
        default=5e-5,
        help="Learning rate for heads (default: 5e-5)",
    )

    # Resume training
    parser.add_argument(
        "--resume-from", type=str, help="GCS checkpoint path to resume from"
    )

    # Training script selection
    parser.add_argument(
        "--script",
        type=str,
        default="scripts/train_cbm_vertex.py",
        help="Training script to use (default: scripts/train_cbm_vertex.py)",
    )

    # Control
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show job configuration without submitting",
    )

    args = parser.parse_args()

    print(f"🧠 CBM Vertex AI Launcher")
    print(f"GPU: {args.gpu_type} {'(preemptible)' if args.preemptible else ''}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}")
    print(f"LR: backbone={args.lr_backbone}, heads={args.lr_heads}")

    # Generate and submit job
    job_name, job_config = generate_job_config(args)
    submit_job(job_name, job_config, args.dry_run)


if __name__ == "__main__":
    main()
