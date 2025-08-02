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

# === INSTALL ===

install:
	pip install -r frontend/requirements.txt -r api/requirements.txt

# === RUN ===

run-backend:
	uvicorn api.fast:app --reload --port 8000

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
	docker build --tag=${IMAGE}:dev .

run_container_local:
	docker run -it -e PORT=8000 -p 8000:8000 ${IMAGE}:dev

# === GCP DEPLOYMENT API ===

build_for_production:
	docker build \
		--platform linux/amd64 \
		-t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod \
		.

push_image_production:
	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod

deploy_to_cloud_run:
	gcloud run deploy \
		--image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod \
		--memory ${MEMORY} \
		--timeout 900 \
		--region ${GCP_REGION} \
		--set-env-vars USE_GCS=true
