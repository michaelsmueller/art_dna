default:
	@echo "Please specify a target to run"

# === HELP ===

help:
	@echo "Usage:"
	@echo "  make install          Install all dependencies"
	@echo "  make run-backend      Run FastAPI backend (http://localhost:8000)"
	@echo "  make run-frontend     Run Streamlit frontend (http://localhost:8501)"
	@echo ""
	@echo "  make create-dataset   Create dataset from raw data (preprocessing)"
	@echo "  make build-dataset    Build image dataset from CSV (VGG16)"
	@echo "  make train-model      Train and save VGG16 model"
	@echo "  make evaluate-model   Evaluate VGG16 model on test set"
	@echo ""
	@echo "  make build_vertex_cloud   Build Vertex AI container via Cloud Build"
	@echo "  make train_vertex_t4      Submit T4 training job (preemptible)"
	@echo "  make train_vertex_a100    Submit A100 training job"
	@echo "  make train_vertex_dry_run Show job config without submitting"

# === INSTALL ===

install:
	pip install -r frontend/requirements.txt -r api/requirements.txt

# === RUN ===

run-backend:
	uvicorn api.fast:app --reload --reload-dir api --port 8000

run-frontend:
	streamlit run frontend/app.py

# === DATA PREPROCESSING ===

create-dataset:
	python model/preprocessing/create_final_df.py

create-artists:
	python model/preprocessing/create_artists.py

# === TRAINING (VGG16 Legacy) ===

build-dataset:
	python model/vgg16-simple/build_dataset.py

train-model:
	python model/vgg16-simple/train_model.py

# === EVALUATE ===

evaluate-model:
	python model/vgg16-simple/evaluate_model.py

# === LOCAL DOCKER API ===

build_container_local:
	docker build -f Dockerfile.local --tag=${IMAGE}:dev .

run_container_local:
	docker run -it -e PORT=8000 -p 8000:8000 ${IMAGE}:dev

# === GCP DEPLOYMENT API ===

build_cloud:
	gcloud builds submit . --config=cloudbuild.yaml --timeout=30m

deploy_to_cloud_run:
	gcloud run deploy \
		--image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod \
		--region ${GCP_REGION} \
		--platform managed \
		--memory ${MEMORY} \
		--cpu 2 \
		--concurrency 1 \
		--timeout 900 \
		--min-instances 1 \
		--max-instances 5 \
		--set-env-vars USE_GCS=true \
		--cpu-boost

# === VERTEX AI TRAINING ===

# Build base image (base dependencies)
build_vertex_base:
	gcloud builds submit . --config=cloudbuild-vertex-base.yaml --timeout=30m

# Build training image (code changes)
build_vertex_cloud:
	gcloud builds submit . --config=cloudbuild-vertex.yaml --timeout=10m

train_vertex:
	python scripts/launch_vertex.py --gpu-type T4

train_vertex_dry_run:
	python scripts/launch_vertex.py --dry-run
