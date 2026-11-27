# Setup & Deployment Guide (Fresh Machine Runbook)

This guide provides the complete step-by-step instructions to set up, train, serve, monitor, and stop the **Customer Churn Prediction MLOps** platform from scratch on a new machine (Linux, macOS / Apple Silicon, or Windows WSL2).

---

## Prerequisites

- **Python**: `3.10`–`3.12` (`3.11` or `3.12` recommended)
- **Package Manager**: [`uv`](https://docs.astral.sh/uv/) (or `pip`)
- **Docker**: Docker Engine `20.10+` & Docker Compose `v2.0+`
- **System Resources**: 8 GB+ RAM (16 GB recommended) and 20 GB+ free disk space
- **Kaggle Account / Credentials** *(only needed if downloading the raw dataset via `kagglehub`)*

---

## 1. Clone & Set Up the Python Environment

From the repository root (`Churn-Prediction-MLOPS/`):

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

> Dependency specification: [requirements.txt](file:///Users/arpitparmar/Projects/MLME%20MLOPS%20Project/Churn%20Prediction%20System/aio2025-mlops-project01/requirements.txt)

---

## 2. Create Docker Network & Environment Files

1. **Create the shared external Docker network:**
   ```bash
   docker network create aio-network
   ```

2. **Create [infra/docker/mlflow/.env](file:///Users/arpitparmar/Projects/MLME%20MLOPS%20Project/Churn%20Prediction%20System/aio2025-mlops-project01/infra/docker/mlflow/.env):**
   ```env
   MYSQL_DATABASE=mlflow
   MYSQL_USER=mlflow
   MYSQL_PASSWORD=mlflow
   MYSQL_ROOT_PASSWORD=root
   AWS_ACCESS_KEY_ID=minio
   AWS_SECRET_ACCESS_KEY=minio123
   ```
   *(Note: `AWS_SECRET_ACCESS_KEY` is also used as `MINIO_ROOT_PASSWORD`, which MinIO requires to be at least 8 characters.)*

3. **Create [infra/docker/airflow/.env](file:///Users/arpitparmar/Projects/MLME%20MLOPS%20Project/Churn%20Prediction%20System/aio2025-mlops-project01/infra/docker/airflow/.env):**
   ```bash
   cp infra/docker/airflow/.env
   ```
   *(On Linux, optionally set `AIRFLOW_UID=$(id -u)` inside `infra/docker/airflow/.env`.)*

4. **Create [serving_pipeline/.env](file:///Users/arpitparmar/Projects/MLME%20MLOPS%20Project/Churn%20Prediction%20System/aio2025-mlops-project01/serving_pipeline/.env):**
   ```bash
   cp serving_pipeline/.env
   ```
   This pre-configures `MODEL_URI=models:/customer_churn_model@champion` and Docker service DNS names (`http://mlflow_server:5000`, `http://minio:9000`, `http://api:8000`).

---

## 3. Start Platform Infrastructure (`mlflow`, `kafka`, `monitor`, `airflow`)

```bash
cd infra/docker
chmod +x run.sh
./run.sh up
./run.sh status
cd ../..
```

> **Platform Notes:**
> - **MinIO Buckets:** `aio_minio_mc` automatically creates both `mlflow` and `churn-mlops` buckets on startup.
> - **GPU Monitoring:** `dcgm-exporter` uses `profiles: ["gpu"]` in [infra/docker/monitor/docker-compose.yaml](file:///Users/arpitparmar/Projects/MLME%20MLOPS%20Project/Churn%20Prediction%20System/aio2025-mlops-project01/infra/docker/monitor/docker-compose.yaml) so it is skipped on macOS/CPU machines. On a Linux NVIDIA GPU host, start it with `docker compose --profile gpu up -d` inside `infra/docker/monitor/`.
> - **Grafana:** Prometheus (`http://prometheus:9090`), Loki (`http://loki:3100`), and the `Churn Prediction & System Overview` dashboard are auto-provisioned via `infra/docker/monitor/grafana/provisioning/`.

---

## 4. Download, Preprocess & Version the Data (`DVC` + `MinIO`)

1. **Download the raw Kaggle dataset** (or place `customer_churn_dataset-training-master.csv` and `customer_churn_dataset-testing-master.csv` into `data-pipeline/data/raw/`):
   ```bash
   python -c 'import kagglehub; print(kagglehub.dataset_download("mohamed444ashraf/customer-churn-data"))'
   ```
   Copy both CSV files from the printed directory into `data-pipeline/data/raw/`.

2. **Generate `processed_churn_data.parquet` and `test.parquet`** using [prepare_feast_data.py](file:///Users/arpitparmar/Projects/MLME%20MLOPS%20Project/Churn%20Prediction%20System/aio2025-mlops-project01/data-pipeline/churn_feature_store/churn_features/feature_repo/prepare_feast_data.py):
   ```bash
   python data-pipeline/churn_feature_store/churn_features/feature_repo/prepare_feast_data.py \
     --input data-pipeline/data/raw/customer_churn_dataset-training-master.csv \
     --output data-pipeline/churn_feature_store/churn_features/feature_repo/data/processed_churn_data.parquet

   python data-pipeline/churn_feature_store/churn_features/feature_repo/prepare_feast_data.py \
     --input data-pipeline/data/raw/customer_churn_dataset-testing-master.csv \
     --output data-pipeline/churn_feature_store/churn_features/feature_repo/data/test.parquet
   ```

3. **Configure your local DVC remote and push datasets to MinIO:**
   ```bash
   cd data-pipeline
   dvc remote modify --local mys3 endpointurl http://localhost:9000
   dvc remote modify --local mys3 access_key_id minio
   dvc remote modify --local mys3 secret_access_key minio123

   dvc add data/raw/customer_churn_dataset-training-master.csv \
           data/raw/customer_churn_dataset-testing-master.csv \
           churn_feature_store/churn_features/feature_repo/data/processed_churn_data.parquet \
           churn_feature_store/churn_features/feature_repo/data/test.parquet
   dvc push
   cd ..
   ```

---

## 5. Initialize Feast Feature Store & Redis Online Store

1. **Start the Feast Redis container:**
   ```bash
   docker run -d -p 6379:6379 --name redis-feast redis:7
   ```

2. **Apply feature definitions and materialize to Redis:**
   ```bash
   cd data-pipeline/churn_feature_store/churn_features/feature_repo
   dvc pull
   feast apply
   feast materialize-incremental "$(date -u +%Y-%m-%dT%H:%M:%S)"
   cd ../../../..
   ```

3. **Verify online feature retrieval:**
   ```bash
   python serving_pipeline/sample_retrieval.py
   ```

---

## 6. Train, Evaluate, Register & Promote the Model (`MLflow`)

From `model_pipeline/src/`:

```bash
cd model_pipeline/src

# 1. Train XGBoost model (logs to MLflow + MinIO; note the printed Run ID)
bash run_sh/train.sh

# 2. Evaluate on holdout test.parquet
bash run_sh/eval.sh --run-id <RUN_ID>

# 3. Register the LoggedModel in MLflow Model Registry (use the training Run ID from step 1, not the eval Run ID)
PYTHONPATH=.. python scripts/register_model.py \
  --config config/config.yaml \
  register \
  --run-id <RUN_ID> \
  --model-name customer_churn_model

# 4. Set @champion alias using the version number printed by step 3 (used by serving_pipeline/.env)
PYTHONPATH=.. python scripts/register_model.py \
  --config config/config.yaml \
  set-alias \
  --model-name customer_churn_model \
  --version <VERSION> \
  --alias champion

cd ../..
```

> **Note on Serving `@champion`:** `serving_pipeline/.env` sets `MODEL_URI=models:/customer_churn_model@champion`, so `churn-prediction-api` automatically loads whichever version has the `@champion` alias on startup. Whenever you promote or reassign `@champion` to a new version, run `docker restart churn-prediction-api` to load it.

> **Promoting Subsequent Challenger Models:**
> Register the new run, set its alias to `staging`, and run:
> ```bash
> cd model_pipeline/src
> bash run_sh/promote.sh --model-name <MODEL_NAME> --version <VERSION> --config config/config.yaml
> ```
> `promote.sh` compares `f1_score` against the current `@champion` and only promotes if it improves.

---

## 7. Build & Start the Serving Pipeline (`FastAPI` + `Gradio UI`)

1. **(Optional, for Drift Monitoring) Generate `reference_data.csv`:**
   cd ../..
   ```bash
   python -c '
   import pandas as pd
   df = pd.read_csv("data-pipeline/data/raw/customer_churn_dataset-training-master.csv").dropna()
   df.columns = [c.replace(" ", "_") for c in df.columns]
   df.to_csv("serving_pipeline/api/data/reference_data.csv", index=False)
   '
   ```

2. **Build and start the API and UI containers:**
   ```bash
   docker compose -f serving_pipeline/docker-compose.yml up -d --build
   ```

3. **Verify startup model pre-loading and health:**
   ```bash
   curl -sL http://localhost:8000/health
   # Expected: {"status":"healthy","model_loaded":true,"timestamp":"..."}
   ```

---

## 8. Service URLs & Credentials

| Service | URL | Credentials |
| :--- | :--- | :--- |
| **Gradio Web UI** | http://localhost:7860 | N/A |
| **FastAPI Swagger Docs** | http://localhost:8000/docs | N/A |
| **FastAPI Prometheus Metrics** | http://localhost:8000/metrics | N/A |
| **Evidently Drift Report** | `http://localhost:8000/monitor/drift?format=html&reference_path=/app/api/data/reference_data.csv&current_path=/app/data_model/production/production.csv` | N/A |
| **MLflow UI** | http://localhost:5000 | N/A |
| **MinIO Console** | http://localhost:9001 | `minio` / `minio123` |
| **Grafana** | http://localhost:3000 | `admin` / `admin` |
| **Prometheus** | http://localhost:9090 | N/A |
| **Airflow UI** | http://localhost:8080 | `airflow` / `airflow` |

---

## 9. Gracefully Stopping & Restarting Services

### Stop Everything (Preserving All Data & Volumes)
```bash
docker compose -f serving_pipeline/docker-compose.yml down
(cd infra/docker && ./run.sh down)
docker stop redis-feast 2>/dev/null || true
docker ps
```

### Restart Later on the Same Machine
```bash
source .venv/bin/activate
(cd infra/docker && ./run.sh up)
docker start redis-feast 2>/dev/null || docker run -d -p 6379:6379 --name redis-feast redis:7
docker compose -f serving_pipeline/docker-compose.yml up -d
```

---

## 10. Quick Troubleshooting

| Issue | Resolution |
| :--- | :--- |
| `dvc pull` warns about missing `train.parquet` or `train_period_1.csv` | Generate and `dvc add` / `dvc push` your own files as shown in **Step 4**, or remove unused `.dvc` pointers from the original repo. |
| `airflow-apiserver` unhealthy (`Can't locate revision`) | Reset the Airflow Postgres volume: `(cd infra/docker/airflow && docker compose down -v && docker compose up -d)`. |
| `curl http://localhost:8000/health` shows `"model_loaded": false` | Ensure `@champion` alias is set on `customer_churn_model` in MLflow (**Step 6**) and `serving_pipeline/.env` points to `http://mlflow_server:5000` and `http://minio:9000`. |
| Grafana panels show `No data` | Send a few `/predict/` requests via Gradio or `curl`, wait 10s for Prometheus to scrape `/metrics`, and refresh Grafana (`Churn Prediction & System Overview`). |
