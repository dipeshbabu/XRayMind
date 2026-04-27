# XRayMind API

XRayMind now includes a lightweight FastAPI service for single-image inference, batch inference, and study-packet generation.

> Research use only. This project is not a medical device and should not be used for clinical diagnosis.

## Run locally

```bash
pip install -r requirements.txt
uvicorn xraymind.api:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Optional API key

Set `XRAYMIND_API_KEY` to require `X-API-Key` on protected endpoints.

```bash
export XRAYMIND_API_KEY=change-me
curl -H "X-API-Key: change-me" http://localhost:8000/health
```

`/health` is open. `/predict`, `/batch-predict`, and `/packet` enforce the key when the environment variable is set.

## Single-image prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: change-me" \
  -F "file=@examples/chest_xray.png" \
  -F "top_k=5" \
  -F "threshold=0.5"
```

The endpoint accepts PNG/JPG and DICOM files when `pydicom` is installed.

## Batch prediction

```bash
curl -X POST http://localhost:8000/batch-predict \
  -H "X-API-Key: change-me" \
  -F "files=@case1.png" \
  -F "files=@case2.dcm"
```

Each case returns either a prediction object or a per-file error.

## Study packet ZIP

```bash
curl -X POST http://localhost:8000/packet \
  -H "X-API-Key: change-me" \
  -F "file=@examples/chest_xray.png" \
  -F "make_pdf=false" \
  --output study_packet.zip
```

## Docker

```bash
docker build -t xraymind-api .
docker run -p 8000:8000 -e XRAYMIND_API_KEY=change-me xraymind-api
```

or:

```bash
XRAYMIND_API_KEY=change-me docker compose up --build
```

## DICOM CLI

```bash
python scripts/dicom_to_png.py \
  --dicom case.dcm \
  --png outputs/dicom/case.png \
  --metadata outputs/dicom/case_metadata.json \
  --redacted outputs/dicom/case_redacted.dcm
```

The redaction helper removes private tags and replaces common direct identifiers. It is not a complete HIPAA de-identification pipeline.

## Audit logs

CLI and API flows can append JSONL audit events without storing raw image data.

CLI example:

```bash
xraymind predict --image case.png --out outputs/prediction.json --audit-log outputs/audit/audit.jsonl
```

API default audit path:

```text
outputs/audit/api.jsonl
```
