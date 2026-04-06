FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app

WORKDIR ${APP_HOME}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        default-libmysqlclient-dev \
        netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY alembic.ini main.py ./
COPY config ./config
COPY data ./data
COPY scripts ./scripts
COPY src ./src

RUN chmod +x ./scripts/wait_for_db.sh

EXPOSE 8000

CMD ["sh", "-c", "./scripts/wait_for_db.sh mysql 3306 && python -m alembic upgrade head && python main.py api --host 0.0.0.0 --port 8000"]
