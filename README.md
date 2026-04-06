# Fitbit Sleep Score System

A full-stack sleep analytics platform that ingests Fitbit-style sleep logs, calculates nightly sleep scores using engineered features and ML pipelines, stores results in MySQL, exposes analytics via FastAPI, and visualizes insights through a Streamlit dashboard.

This project simulates a real-world wearable data pipeline including data ingestion, feature engineering, scoring, analytics APIs, and an interactive dashboard.

## Architecture Overview

```text
Fitbit Sleep Data
        |
        v
Data Processing Layer
(src/data)
        |
        v
Feature Engineering
        |
        v
Sleep Scoring Models
(src/models)
        |
        v
FastAPI Backend
(src/api)
        |
        v
MySQL Database
(src/database)
        |
        v
Streamlit Analytics Dashboard
(streamlit_app)
```

## Project Structure

```text
fitbit-sleep-score
|
+-- src
|   +-- api              # FastAPI routes and schemas
|   +-- data             # Fitbit ingestion + feature engineering
|   +-- database         # SQLAlchemy models + CRUD + migrations
|   \-- models           # Sleep scoring and anomaly detection
|
+-- streamlit_app        # Interactive analytics dashboard
|
+-- docker               # Dockerfiles and MySQL configuration
|
+-- data/sample          # Synthetic data generation utilities
|
+-- notebooks            # Jupyter experimentation environment
|
+-- tests                # API, data, model and visualization tests
|
+-- docker-compose.yml
+-- docker-compose.dev.yml
+-- docker-compose.test.yml
|
\-- main.py              # CLI entrypoint
```

## Key Features

### Sleep Data Ingestion

Processes Fitbit-style sleep logs including:

- sleep stages
- time in bed
- heart rate
- HRV
- oxygen saturation

### Feature Engineering

Transforms raw logs into nightly features such as:

- sleep efficiency
- deep/REM distribution
- sleep fragmentation
- recovery indicators

### Sleep Score Engine

Computes a composite sleep score based on:

- duration
- efficiency
- sleep stages
- physiological metrics

### FastAPI Backend

Provides REST APIs for:

- sleep analysis
- nightly reports
- trend analytics
- recommendations
- model metadata

### Streamlit Dashboard

Interactive UI for:

- nightly sleep score
- stage visualization
- trend analysis
- sleep recommendations
- weekly analytics

### Dockerized Stack

Containerized services:

- MySQL database
- FastAPI backend
- Streamlit dashboard
- optional JupyterLab environment

## Technology Stack

### Backend

- Python 3.12
- FastAPI
- SQLAlchemy
- Alembic
- MySQL
- Pydantic

### Data / ML

- PyTorch
- Pandas
- NumPy

### Frontend

- Streamlit

### DevOps

- Docker
- Docker Compose

## Setup Instructions

### Prerequisites

- Python 3.12
- Docker Desktop (recommended)
- MySQL 8 (optional for local setup)

### 1. Create Virtual Environment

```powershell
python -m venv .venv
```

Activate:

Windows:

```powershell
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

### 2. Install Dependencies

Backend dependencies:

```powershell
pip install -r requirements.txt
```

Streamlit dashboard dependencies:

```powershell
pip install -r requirements.streamlit.txt
```

Full development environment:

```powershell
pip install -r requirements.dev.txt
```

### 3. Configure Environment

Review configuration:

`config/config.yaml`

Follow setup guide:

`SECRETS_SETUP.md`

Set environment variables such as:

- `DATABASE_URL`
- `API_BASE_URL`

### 4. Run Database Migrations

```powershell
python -m alembic upgrade head
```

### 5. Start FastAPI Server

```powershell
python main.py api
```

API will start at:

`http://localhost:8000`

### 6. Start Streamlit Dashboard

```powershell
streamlit run streamlit_app/app.py
```

Dashboard will be available at:

`http://localhost:8501`

## Docker Setup (Recommended)

Run the full stack:

```bash
docker compose up --build
```

Services started:

| Service | Port |
| --- | --- |
| FastAPI | 8000 |
| Streamlit | 8501 |
| JupyterLab | 8888 |
| MySQL | 3306 |

## API Documentation

### Swagger UI

`http://localhost:8000/docs`

### OpenAPI JSON

`http://localhost:8000/openapi.json`

## Authentication

Most endpoints require the header:

`X-User-Id`

Register a user first.

### Register User Example

`POST /auth/register`

Example payload:

```json
{
  "fitbit_user_id": "demo_user",
  "email": "demo@example.com",
  "full_name": "Demo User",
  "birth_date": "1995-01-01",
  "sex": "male",
  "timezone": "UTC"
}
```

### Analyze Sleep Example

`POST /sleep/analyze`

Headers:

`X-User-Id: 1`

Example payload:

```json
{
  "date": "2026-04-06",
  "fitbit_log_id": 1,
  "start_time": "2026-04-05T23:00:00",
  "end_time": "2026-04-06T06:00:00",
  "time_in_bed": 420,
  "minutes_asleep": 400,
  "minutes_awake": 20,
  "awakenings_count": 2,
  "efficiency": 95,
  "hrv": 60,
  "resting_hr": 58,
  "spo2": 98,
  "stages": []
}
```

## Key API Endpoints

| Endpoint | Description |
| --- | --- |
| `POST /auth/register` | Register user |
| `GET /auth/me` | Current user |
| `POST /sleep/analyze` | Analyze sleep data |
| `GET /sleep/score/{date}` | Nightly sleep score |
| `GET /sleep/trend` | Trend analysis |
| `GET /sleep/stages/{date}` | Sleep stage intervals |
| `GET /sleep/recommendations/{date}` | Sleep recommendations |
| `GET /analytics/weekly` | Weekly analytics |
| `GET /analytics/best-night` | Best sleep night |
| `GET /analytics/worst-night` | Worst sleep night |

## Testing

Run tests:

```powershell
pytest
```

Tests cover:

- API endpoints
- data pipeline
- feature engineering
- sleep scoring models
- visualization logic

## Deployment

The project supports deployment to:

- Railway
- Render
- Fly.io
- AWS ECS
- Kubernetes
- Azure Container Apps

Using:

`docker/Dockerfile`

Recommended production setup:

- managed MySQL database
- environment secrets
- separate Streamlit service

## Future Improvements

- Fitbit API integration
- personalized ML sleep models
- long-term sleep health analytics
- mobile dashboard
- real-time sleep tracking

## Author

**Amit Bidlane** 
Python Developer | Backend Development | Data Systems
