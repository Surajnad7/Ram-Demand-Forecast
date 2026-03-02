#!/usr/bin/env bash
# deploy.sh — Build & deploy to GCP Cloud Run
# Usage: ./deploy.sh <PROJECT_ID> [REGION]
set -euo pipefail

PROJECT_ID="${1:?Usage: ./deploy.sh PROJECT_ID [REGION]}"
REGION="${2:-us-central1}"
SERVICE="ram-demand-forecast"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}"

echo "▸ Building image …"
gcloud builds submit --tag "${IMAGE}" --project "${PROJECT_ID}"

echo "▸ Deploying to Cloud Run …"
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --port 8080 \
  --project "${PROJECT_ID}"

echo "▸ Setting up weekly Monday retrain via Cloud Scheduler …"
SERVICE_URL=$(gcloud run services describe "${SERVICE}" \
  --region "${REGION}" --project "${PROJECT_ID}" \
  --format='value(status.url)')

gcloud scheduler jobs create http "${SERVICE}-weekly-retrain" \
  --schedule="0 6 * * 1" \
  --uri="${SERVICE_URL}/train" \
  --http-method=POST \
  --location="${REGION}" \
  --project="${PROJECT_ID}" \
  --attempt-deadline=600s \
  2>/dev/null || echo "  (scheduler job already exists)"

echo "✓ Deployed → ${SERVICE_URL}"
