# CarErescue — OBD Diagnostics Service

A **standalone** FastAPI microservice for OBD-II vehicle data, separate from the
main CarErescue backend (own config, own `.env`, own port). Built to deploy on
**Railway** via the bundled `Dockerfile`.

It runs an **anomaly-detection** model: per-vehicle baselining + robust
normalization + windowed scoring, returning a diagnosis (anomaly score + the
likely affected system). It only **returns** the diagnosis — it does not call
back into the CarErescue API.

## Dataset

Real **Toyota Etios 2014** OBD-II recordings from
[`eron93br/carOBD`](https://github.com/eron93br/carOBD) (Carloop logger, 1 Hz,
27 PIDs). A representative subset is bundled in `app/data/etios/`
(`drive*`, `idle*`, `live*`). Add more CSVs to that folder to extend it.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Service banner |
| GET | `/health` | Health check (Railway healthcheck path) |
| GET | `/dataset/info` | Vehicle, files, total rows, column count |
| GET | `/dataset/files` | List bundled CSV files |
| GET | `/dataset/columns` | Column → field → unit → PID schema |
| GET | `/dataset/sample?file=&n=` | N rows as `OBDSnapshot` objects |
| GET | `/dataset/stats?file=` | Per-column min/max/mean/std |
| POST | `/diagnostics/analyze` | Score one `OBDSnapshot` (quick, transient-sensitive) |
| POST | `/diagnostics/analyze-window` | Score a window of recent snapshots (**recommended**) |
| POST | `/diagnostics/baseline/{vehicle_id}` | Register a car's own baseline from healthy snapshots |
| GET | `/diagnostics/status` | Model metadata + validation numbers |
| GET | `/docs` | Swagger UI |

## How the anomaly detector works

1. **Per-vehicle baseline** — each car's healthy readings define its own
   robust centre/spread (median + MAD) per feature. Readings are converted to
   robust z-scores, so "abnormal" means the same thing on any car (this is what
   makes it generic). A car with no baseline falls back to a population baseline
   built from the bundled Etios data.
2. **Health vs context features** — only *health-indicator* signals (coolant,
   fuel trims, voltage, catalyst, AFR) drive the score. Driver-controlled
   signals (RPM, throttle, load, speed) are context, never "faults."
3. **Windowed scoring** — `/analyze-window` median-aggregates recent readings to
   remove transients, so only *sustained* abnormal conditions flag.
4. **Score + system** — the anomaly score (0–1) is the largest robust health
   z-score; an IsolationForest adds a secondary multivariate vote. The strongest
   deviating features are mapped to a likely system (a CarErescue `SkillName`).

### ⚠️ Honest note on the bundled data
The Etios CSVs are a **single, unlabeled car** and the raw PIDs are
**mis-scaled** (a known Carloop logging artifact), with warmup/ambient drift.
So the validation numbers in `/diagnostics/status` are **illustrative of the
pipeline, not a production benchmark** — the shipped threshold is deliberately
**conservative** (reliably catches *severe* sustained anomalies; subtler ones
need cleaner data). On properly-decoded real OBD data with a fresh per-vehicle
baseline, sensitivity is substantially better. Retrain anytime with
`python -m app.ml.train`.

## Run locally

```bash
cd obd_service
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
# open http://localhost:8001/docs
```

## Run with Docker

```bash
docker build -t obd-service .
docker run -p 8001:8001 obd-service
```

## Deploy on Railway

1. New Project → Deploy from repo, set **root directory** to `obd_service`.
2. Railway detects the `Dockerfile` (and `railway.json`).
3. Railway injects `PORT` automatically — the app binds to it.
4. Set any variables from `.env.example` in the **Variables** tab.
5. Healthcheck path is `/health`.

## The OBD snapshot contract

`app/schemas.py` defines `OBDSnapshot` = `{ vehicle_id, timestamp, source,
health, sensors }`. Every sensor field is optional (real dongles don't expose
every PID). This is the unit the future anomaly-detection pipeline will consume.
