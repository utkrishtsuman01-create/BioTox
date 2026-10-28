# BioTox — Chemical Intelligence

BioTox is an educational front-end prototype for a web-based chemical toxicity screening platform. It demonstrates the user journey from chemical input to a structured report in a polished, responsive interface.

## Current status

**Front-end prototype only. No trained machine-learning model is connected.** The interactive report generates illustrative interface data so the workflow can be demonstrated; it is not a scientific toxicity prediction and must not be used for safety decisions.

## Features

- Responsive dark scientific dashboard
- Compound-name and SMILES input modes
- Endpoint and target-context selectors
- Illustrative result panel with clear demo-only labeling
- Exportable JSON demo report
- Workflow overview, platform features, FAQ, and responsible-use caveats
- No backend or API keys required for the current static prototype

## Run locally

Open `index.html` in a modern browser. For local development, you may also run a static server:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000`.

## Optional Tox21 baseline training pipeline

A reproducible starter pipeline is provided in `backend/train_tox21.py`. It downloads the public Tox21 benchmark, validates SMILES with RDKit, builds Morgan fingerprints, trains one Random Forest per assay endpoint, and writes model artifacts plus an evaluation report. This pipeline is a starting point for a college project, not a scientifically qualified safety model.

Run from the repository root after installing `backend/requirements.txt`:

```bash
cd backend
python train_tox21.py
pytest -q
```

The Tox21 dataset contains assay-specific labels. The models estimate the probability of a positive assay label; this must not be described as a universal toxicity probability. The included random stratified split is only a baseline and can overestimate generalization because related chemical structures can cross splits. Before scientific use, add scaffold-based and external validation, endpoint-level review, calibration, applicability-domain checks, and documented dataset provenance.

The optional `POST /api/v1/tox21/predict` endpoint only runs when a trained artifact for the requested assay exists. Without one, it returns a clear HTTP 503 response rather than fabricating a prediction. The existing browser-only interface remains explicitly marked as a demo and does not yet call the API.

## Suggested architecture for real ML predictions

1. **Chemistry parsing:** validate submitted SMILES and normalize molecular structures with RDKit.
2. **Feature engineering:** generate the exact molecular descriptors/fingerprints used during model training.
3. **Model training:** document dataset provenance, endpoint definitions, label quality, train/validation/test splits, and class imbalance handling.
4. **Evaluation:** report appropriate metrics (for example ROC-AUC, PR-AUC, sensitivity, specificity, calibration), external validation, and limitations.
5. **API:** build a Python API (for example FastAPI) that validates requests, loads a versioned model, computes features, and returns structured predictions.
6. **Frontend integration:** replace the demo-only JavaScript handler with a request to the API; show model version, endpoint, uncertainty/calibration where supported, and applicability-domain warnings.
7. **Responsible use:** do not present model scores as proof of safety. Include provenance, intended use, data limitations, and a clear statement that predictions do not replace laboratory testing or professional toxicological assessment.

## Example response contract for a future API

```json
{
  "compound": "submitted identifier",
  "endpoint": "defined endpoint",
  "prediction": "model-defined category",
  "probability": null,
  "model_version": "version identifier",
  "applicability_domain": "in_domain | uncertain | out_of_domain",
  "limitations": ["Model- and dataset-specific caveats"]
}
```

The API must return values produced by a real, evaluated model. Do not use placeholder or randomly generated values in a production prediction endpoint.

## Important disclaimer

BioTox is an educational research prototype, not a medical device, regulatory tool, safety certification, or substitute for validated laboratory testing, exposure assessment, or qualified toxicological judgment.
