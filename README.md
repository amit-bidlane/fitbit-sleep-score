# Fitbit Sleep Score System

Fitbit Sleep Score System is a Python application for ingesting Fitbit-style sleep data, engineering nightly features, calculating sleep scores, persisting results in MySQL, serving analytics through FastAPI, and visualizing trends in Streamlit.

## Project Architecture

The project is organized into five main layers:

- `src/data`: Fitbit ingestion, preprocessing, and feature engineering.
- `src/database`: SQLAlchemy models, async DB session management, CRUD operations, and Alembic migrations.
- `src/models`: Sleep scoring, anomaly detection, and PyTorch training utilities.
- `src/api`: FastAPI schemas and routes for analysis, reporting, trend views, and user registration.
- `streamlit_app`: Multi-page dashboard and API client for the user-facing analytics experience.

Supporting directories:

- `docker`: Dockerfiles plus MySQL init/config assets.
- `data/sample`: Synthetic data generation utilities.
- `notebooks`: Jupyter workspace for experiments.
- `tests`: API, data, model, CRUD, and visualization coverage.

## Setup Instructions

### 1. Prerequisites

- Python 3.12 is recommended.
- MySQL 8.x for local non-Docker setup.
- Docker Desktop if you want to use Compose.

### 2. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Core backend:

```powershell
pip install -r requirements.txt
```

Dashboard and API client extras:

```powershell
pip install -r requirements.streamlit.txt
```

Full development toolchain:

```powershell
pip install -r requirements.dev.txt
```

### 4. Configure secrets and settings

- Review `config/config.yaml` for non-secret defaults.
- Follow `SECRETS_SETUP.md` for environment variables and Streamlit secrets.
- Set `DATABASE_URL` if you are not using the Docker default host name `mysql`.

### 5. Initialize the database

Run migrations:

```powershell
python -m alembic upgrade head
```

### 6. Start the API

```powershell
python main.py api --host 0.0.0.0 --port 8000
```

### 7. Start the Streamlit dashboard

```powershell
streamlit run streamlit_app/app.py
```

## API Usage

The FastAPI app is served from `src.api.routes:app`.

Interactive docs:

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

Authentication model:

- Most endpoints require the `X-User-Id` header.
- Register a user first with `POST /auth/register`.

### Common endpoints

- `POST /auth/register`
- `GET /auth/me`
- `POST /sleep/analyze`
- `GET /sleep/score/{target_date}`
- `GET /sleep/trend?days=14`
- `GET /sleep/stages/{target_date}`
- `GET /sleep/recommendations/{target_date}`
- `GET /sleep/report/{target_date}`
- `GET /analytics/weekly`
- `GET /analytics/best-night`
- `GET /analytics/worst-night`

### Example: register a user

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "fitbit_user_id": "demo-user",
    "email": "demo@example.com",
    "full_name": "Demo User"
  }'
```

### Example: analyze a night

```bash
curl -X POST http://localhost:8000/sleep/analyze \
  -H "Content-Type: application/json" \
  -H "X-User-Id: 1" \
  -d '{
    "date": "2026-04-05",
    "start_time": "2026-04-05T22:00:00",
    "end_time": "2026-04-06T06:00:00",
    "time_in_bed": 480,
    "minutes_asleep": 450,
    "minutes_awake": 30,
    "minutes_after_wakeup": 5,
    "minutes_to_fall_asleep": 10,
    "awakenings_count": 1,
    "efficiency": 93.8,
    "hrv": 55.0,
    "resting_hr": 57.0,
    "spo2": 97.0,
    "stages": []
  }'
```

## Docker Setup

The repository includes:

- `docker/Dockerfile` for FastAPI and migrations.
- `docker/Dockerfile.streamlit` for the dashboard.
- `docker/Dockerfile.jupyter` for notebook workflows.
- `docker-compose.yml` for the standard multi-service stack.
- `docker-compose.dev.yml` for a bind-mounted development workflow.
- `docker-compose.test.yml` for running tests against MySQL.

### Start the full stack

```bash
docker compose up --build
```

This starts:

- `mysql`
- `fastapi`
- `streamlit`
- `jupyter`

### Development mode

```bash
docker compose -f docker-compose.dev.yml up --build
```

### Test mode

```bash
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit
```

### Default service URLs

- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`
- JupyterLab: `http://localhost:8888`
- MySQL: `localhost:3306`

## Deployment Guide

### Railway

This repo includes `railway.toml` configured to build from `docker/Dockerfile`, wait for MySQL, run Alembic migrations, and then launch FastAPI.

Recommended Railway setup:

1. Provision a MySQL service.
2. Set `DATABASE_URL` with the Railway MySQL connection string.
3. Set Fitbit secrets from `SECRETS_SETUP.md`.
4. Deploy the repo and confirm the health check at `/docs`.

### Generic container platforms

For Render, Fly.io, Azure Container Apps, ECS, or Kubernetes:

1. Build from `docker/Dockerfile` for the API service.
2. Inject `DATABASE_URL` and Fitbit secrets as environment variables.
3. Run `python -m alembic upgrade head` during release or startup.
4. Expose the API on the platform-provided port.
5. Deploy Streamlit separately with `docker/Dockerfile.streamlit` if you want the dashboard hosted independently.

### Production notes

- Use managed MySQL instead of the bundled Compose database.
- Store secrets in the platform secret manager, not in `config/config.yaml`.
- Keep the Streamlit `API_BASE_URL` pointed at the deployed FastAPI base URL.
- Review Fitbit API lifecycle constraints before committing to long-term production use.
