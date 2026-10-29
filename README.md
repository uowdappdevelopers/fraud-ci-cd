# fraud-ci-cd

A card-fraud scoring service with an automated pipeline (MAAI 903, Lesson 6 / Workshop 6).

Every push and pull request runs `.github/workflows/ci.yml`:

1. **Lint** the code (`ruff`)
2. **Validate the data** (`ci/validate_data.py`)
3. **Train** the model (`train.py`)
4. **Quality gate**: block a model that is worse than the approved baseline (`ci/quality_gate.py`)
5. **Test** the API, the training code, the data checks and the gate (`pytest`)
6. **Build** the Docker image, start it, **smoke-test** it, and push it to GitHub Container Registry
   (only from `main` or a version tag)

`Jenkinsfile` holds the same pipeline for Jenkins.

## Run the same checks on your own machine

    pip install -r requirements.txt -r requirements-dev.txt
    ruff check .
    python ci/validate_data.py data/transactions.csv
    python train.py --data data/transactions.csv --out model
    python ci/quality_gate.py
    python -m pytest -q

Or run all `run:` steps of the workflow in order: `python ci/run_local.py`

## Files

| Path | Purpose |
|---|---|
| `app.py` | the REST service (`/health`, `/metadata`, `/metrics`, `/predict`, `/predict_batch`) |
| `train.py` | trains the model and writes `model/` and `metrics.json` |
| `ci/validate_data.py` | data checks before training |
| `ci/quality_gate.py` | rules a new model must pass (floors, baseline, per-group recall) |
| `baseline/metrics.json` | scores of the approved model; update it only through a reviewed pull request |
| `ci/smoke_test.py` | five checks against a running service |
| `ci/run_local.py` | runs a workflow job's `run:` steps locally |
| `Dockerfile` | the image: base, pinned libraries, code, model, non-root user, health check |
| `.github/workflows/ci.yml` | the GitHub Actions pipeline |
| `.github/dependabot.yml` | weekly pull requests for new action and package versions |
