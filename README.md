# Real-Time Network Intrusion Detection System (NIDS) — MLOps Pipeline

XGBoost + SMOTE classifier on CICIDS-2017, served in real time via FastAPI,
explained with SHAP, monitored for drift with Evidently AI, tracked with
MLflow, and auto-retrained when drift is detected.

Built for the CIT MLOps project (Rakshana P, Niranjana Sree G — 24AM0086, 24AM0072).

---

## 1. What's in this repo

```
network-ids-mlops/
├── config/                     (reserved for future YAML configs)
├── data/
│   ├── raw/                    ← put the 8 CICIDS-2017 CSVs here
│   └── processed/              ← train/test/reference parquet files (generated)
├── src/
│   ├── config.py                Central paths, hyperparameters, thresholds
│   ├── data/
│   │   ├── preprocess.py        Loads raw CSVs → cleans → groups labels → train/test split
│   │   └── load_data.py         Loads processed data, label-encodes, applies SMOTE
│   ├── models/
│   │   └── train.py             Trains XGBoost, logs to MLflow, registers/promotes model
│   ├── explainability/
│   │   └── shap_explain.py      SHAP TreeExplainer wrapper (per-prediction top features)
│   ├── monitoring/
│   │   └── drift_monitor.py     Evidently AI drift report (reference vs live window)
│   └── api/
│       ├── main.py              FastAPI app: /health, /predict
│       └── schemas.py           Pydantic request/response models
├── scripts/
│   ├── generate_sample_data.py  Synthetic CICIDS-2017-shaped data (for plumbing tests)
│   ├── csv_streamer.py          Replays test-set rows against the live API (simulates traffic)
│   └── retrain_trigger.py       Reads drift status → retrains if drift detected
├── dashboard/
│   └── app.py                   Streamlit dashboard (MLflow runs, drift, live predictions)
├── docker/
│   ├── Dockerfile               API container
│   └── docker-compose.yml       API + MLflow UI + Dashboard, all together
├── .github/workflows/ci-cd.yml  pytest → docker build → DockerHub push
├── tests/test_api.py            API + preprocessing tests
└── requirements.txt
```

Every module above already runs — I tested the full loop (data generation →
preprocess → train → serve → stream traffic → drift check → retrain-trigger
→ dashboard) before handing this back to you.

---

## 2. Set up your environment (VS Code, ASUS TUF A15)

You don't need your GPU for this — XGBoost trains fine on CPU, and Evidently/SHAP/FastAPI are all CPU-only anyway.

1. Install **Python 3.11** (3.12 also works) and the **Python + Pylance** VS Code extensions.
2. Open the `network-ids-mlops/` folder in VS Code (`File → Open Folder`).
3. Open a terminal in VS Code (`` Ctrl+` ``) and create a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS/Linux
   ```

4. In VS Code, select this venv as your interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → pick `.venv`.
5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Get the real dataset (do this once)

1. Download **CICIDS-2017** from Kaggle: search "CIC-IDS-2017" (e.g. the `cicids2017` dataset by `cicdataset` or the official [UNB CIC page](https://www.unb.ca/cic/datasets/ids-2017.html)).
2. You'll get 8 CSVs (Monday–Friday captures, ~6GB total, 2.8M+ rows). Unzip them straight into `data/raw/`:

   ```
   data/raw/Monday-WorkingHours.pcap_ISCX.csv
   data/raw/Tuesday-WorkingHours.pcap_ISCX.csv
   data/raw/Wednesday-workingHours.pcap_ISCX.csv
   data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   data/raw/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   data/raw/Friday-WorkingHours-Morning.pcap_ISCX.csv
   data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
   data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   ```

3. That's it — `src/data/preprocess.py` reads every CSV in that folder automatically, whatever the filenames.

**Don't have the download yet?** Skip to Step 4's synthetic option — you can test 100% of the pipeline today and swap in real data later without changing any code.

---

## 4. Run the pipeline

### Option A — quick plumbing test with synthetic data (do this first)

```bash
python -m scripts.generate_sample_data
python -m src.data.preprocess
python -m src.models.train
```

This proves every stage works before you commit to the ~10–20 minute run on
the real 2.8M-row dataset. Expect low, meaningless accuracy here — it's
random synthetic noise, not real traffic (this is stated explicitly in the
script). Delete `data/raw/SAMPLE-synthetic.csv` before running Option B.

### Option B — the real thing

```bash
python -m src.data.preprocess     # ~2-5 min: cleans 2.8M rows, groups labels, splits
python -m src.models.train        # trains XGBoost + SMOTE, logs to MLflow, saves model
```

You should see a classification report print in the terminal, and a target
Macro-F1 around 0.92+ (per your project's stated goal) once trained on the
full real dataset — synthetic data will not hit this.

### Inspect experiments in MLflow

```bash
mlflow ui --backend-store-uri file:./mlruns
```
Open http://127.0.0.1:5000 — every run's params, metrics, confusion matrix, and registered model versions live here.

### Serve real-time predictions

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```
Open http://127.0.0.1:8000/docs for interactive Swagger UI. Try `/health` first, then `/predict` with a body like:
```json
{ "features": { "Flow Duration": 1000, "Total Fwd Packets": 5, "...": 0 } }
```
Any missing feature keys default to 0 — you don't need all ~26 columns for a quick test.

### Simulate live network traffic

With the API running in one terminal, in another:
```bash
python -m scripts.csv_streamer --n 600 --delay 0.05
```
This replays 600 rows from your held-out test set through `/predict`, one at a time — standing in for a real packet-capture feed. Every prediction gets logged to `logs/predictions_log.jsonl`, which is what drift monitoring reads next.

### Check for drift

```bash
python -m src.monitoring.drift_monitor
```
Compares the last 500 logged predictions against the training-time reference distribution. Writes `logs/drift_report.html` (open it directly in a browser) and `logs/drift_status.json`.

### Auto-retrain if drift was detected

```bash
python -m scripts.retrain_trigger
```
Reads `drift_status.json`; if drift crossed the threshold, retrains and only promotes the new model to "Production" in the MLflow registry if its F1-macro is at least as good as the current one. This is the self-healing loop from your architecture diagram.

### Dashboard

```bash
streamlit run dashboard/app.py
```
Open http://127.0.0.1:8501 — shows MLflow run history, drift status with the full Evidently report embedded, and a live feed of recent predictions.

---

## 5. Run with Docker

```bash
# from the project root, after you've trained a model locally at least once
docker compose -f docker/docker-compose.yml up --build
```
This starts three containers: the API (`:8000`), MLflow UI (`:5000`), and the dashboard (`:8501`). Model artifacts and logs are mounted as volumes from your local `models/` and `logs/` folders (not baked into the image), so retraining locally is picked up without rebuilding.

---

## 6. Tests

```bash
pytest tests/ -v
```
Covers the `/health` and `/predict` endpoints, malformed-payload handling, and preprocessing helpers. `test_predict_*` tests auto-skip if you haven't trained a model yet.

---

## 7. CI/CD

`.github/workflows/ci-cd.yml` runs on every push/PR to `main`: installs deps, generates synthetic data (CI has no access to the real dataset), runs the full pipeline, then runs `pytest`. The DockerHub push step is disabled by default — enable it by setting a repo variable `DOCKERHUB_ENABLED=true` and adding `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` secrets in your GitHub repo settings.

---

## 8. Known warnings (safe to ignore)

- `FutureWarning` about MLflow's `stages` API being deprecated in favor of aliases — still fully functional, just an older API surface. Fine to leave as-is for a course project.
- Pydantic `"model_loaded"`/`"model_version"` protected-namespace warnings — cosmetic only.
- `on_event("startup")` deprecation in FastAPI — still works; a future refactor could move to `lifespan` handlers.

---

## 9. Suggested demo flow for your presentation

1. Show `mlflow ui` with the run history and confusion matrix artifact.
2. Show Swagger docs (`/docs`) and fire a live `/predict` request, pointing out the SHAP `top_contributing_features` in the response — this is your "explainable alerts" claim from the deck.
3. Run `csv_streamer.py`, then `drift_monitor.py` live, and open the Evidently HTML report.
4. Show the Streamlit dashboard tying it all together visually.
5. Optionally: manually corrupt/shift a chunk of streamed traffic (e.g. scale a few columns) to force `drift_detected: true`, then run `retrain_trigger.py` live to demonstrate the self-healing loop.
