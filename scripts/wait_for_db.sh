#!/usr/bin/env sh

set -eu

HOST="${1:-mysql}"
PORT="${2:-3306}"
TIMEOUT="${DB_WAIT_TIMEOUT:-60}"
STARTED_AT="$(date +%s)"

echo "Waiting for database at ${HOST}:${PORT}..."

while ! nc -z "${HOST}" "${PORT}" >/dev/null 2>&1; do
    NOW="$(date +%s)"
    ELAPSED="$((NOW - STARTED_AT))"
    if [ "${ELAPSED}" -ge "${TIMEOUT}" ]; then
        echo "Timed out waiting for database after ${TIMEOUT}s."
        exit 1
    fi
    sleep 2
done

echo "Database is reachable."
