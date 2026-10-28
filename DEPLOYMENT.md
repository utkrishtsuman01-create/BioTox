# BioTox deployment guide

## Frontend on Vercel

Import `utkrishtsuman01-create/BioTox` into Vercel and keep the project root at `./`. The root is a static HTML website; no build command or environment variables are required. Test the home page and favicon on desktop and mobile after deployment.

## Backend API

FastAPI lives in `backend/`. Deploy it separately to a Python-container-capable host using `backend/Dockerfile`. Set `BIOTOX_ALLOWED_ORIGINS` to the exact frontend origin(s); do not use a wildcard in production.

- Health: `GET /health`
- API docs: `/docs`
- Prediction scaffold: `POST /api/v1/predict`

Prediction currently returns HTTP 503 with `MODEL_NOT_CONFIGURED`. This is intentional: no trained model or validated dataset is included. The API must not invent scientific predictions. Connect a real model only after documenting its endpoint, dataset provenance, validation, uncertainty and applicability-domain behavior.

## Local development

Frontend: open `index.html` or run `python -m http.server 8000` from the repository root.

Backend:

```bash
cd backend
python -m venv .venv
pip install -r requirements.txt
uvicorn app:app --reload
pytest -q
```

## Before a scientific release

- Use a documented, legally usable dataset with explicit endpoint and label provenance.
- Avoid train/test leakage and use an external test set where possible.
- Validate exact molecular featurization and model version.
- Report endpoint-appropriate metrics, uncertainty/calibration, and applicability domain.
- Never claim a prediction establishes that a chemical is safe.
