# SkyGuard AI

**AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations (AWS)**
Smart India Hackathon 2026 · Problem Statement SIH26073 · Theme: Disaster Management · Team Nexus

SkyGuard AI detects anomalous, inconsistent, or faulty AWS sensor readings in
real time using **only** Temperature (°C), Atmospheric Pressure (hPa), and
Relative Humidity (%) — and, critically, tries to tell a genuine severe-weather
event apart from a sensor/data fault, explains *why* it flagged something, and
can optionally self-heal the corrupted reading.

This is a **software-only** project. No ESP32/Arduino/Raspberry Pi or physical
hardware is required or used anywhere in this codebase.

---

## How it works — 6-module pipeline

```
AWS DATA STREAM (T, P, RH)
        │
        ▼
Module 1 — Temporal & Seasonal Baseline
   Learns each station's normal diurnal (hour-of-day) pattern per parameter.
        │
        ▼
Module 2 — Multivariate Consistency (VPD / Psychrometric check)
   Uses Tetens/Magnus vapor-pressure equations to check whether the T, P, RH
   triple is physically co-possible (core innovation — not just per-channel
   thresholding).
        │
        ▼
Anomaly Score Fusion → is_flagged
        │
        ▼
Module 3 — Lightweight Spatial Check
   Compares a flagged station against its 3-5 nearest neighbors (precomputed
   Haversine distance matrix, no external GIS/network dependency) at the same
   timestamp: do neighbors agree (→ genuine weather event) or disagree
   (→ isolated sensor fault)?
        │
        ▼
Module 4 — Root-Cause Classifier
   Gradient-boosted trees (LightGBM if installed, else scikit-learn) classify
   flagged readings: spike / frozen sensor / comm error / calibration drift /
   genuine event.
        │
        ▼
Module 5 — Confidence Score + Explainable AI (SHAP)
   Produces a plain-English narrative of *why* a reading was flagged. Falls
   back to a feature-importance explanation if the `shap` package isn't
   installed, so the pipeline never breaks.
        │
        ▼
Module 6 — Optional Self-Healing
   Cross-parameter / temporal imputation for readings judged to be sensor
   faults. NEVER a silent overwrite: every imputed row is flagged
   `imputed=True` and the raw original value is preserved in `<param>_original`.
        │
           ▼
         Web command center (Next.js): alerts, severity, sensor health, corrected values
```

## Project structure

```
SkyGuard-AI/
├── config.yaml                 # all tunable thresholds/parameters (nothing hard-coded in source)
├── requirements.txt
├── train.py                    # trains the Module 4 root-cause classifier
├── evaluate.py                 # runs the full pipeline + reports detection metrics
├── data/
│   └── sample/                 # generated synthetic datasets (created by scripts/)
├── models/                     # trained classifier is saved here
├── src/
│   ├── config.py                # YAML config loader
│   ├── data_generator.py        # synthetic AWS network + known-fault injector
│   ├── baseline.py               # Module 1
│   ├── physics.py                # Module 2 (VPD/psychrometrics)
│   ├── anomaly.py                 # anomaly score fusion, health score, fault evolution
│   ├── spatial.py                 # Module 3 (Haversine spatial cross-check)
│   ├── classifier.py               # Module 4 (root-cause classifier)
│   ├── explainability.py            # Module 5 (SHAP / fallback explanations)
│   ├── self_healing.py               # Module 6
│   └── pipeline.py                    # orchestrates all 6 modules end-to-end
├── dashboard/
│   └── app.py                   # legacy Streamlit dashboard
├── backend/
│   └── api.py                   # FastAPI adapter over the existing pipeline
├── web/
│   ├── app/                     # Next.js command-center frontend
│   ├── package.json
│   └── tailwind.config.ts
├── scripts/
│   ├── generate_sample_data.py   # synthetic demo network generator
│   └── inject_known_faults.py    # labeled evaluation dataset generator
└── tests/
    └── test_pipeline.py          # smoke tests for every module
```

## Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

`lightgbm` and `shap` are recommended but optional — the project degrades
gracefully to scikit-learn's `GradientBoostingClassifier` and a
feature-importance-based explanation respectively if either is unavailable
(tested in this environment, which had no network access to install them).

## Quickstart — demo mode (no real AWS dataset needed)

```bash
python scripts/setup_environment.py
python scripts/generate_sample_data.py
python scripts/inject_known_faults.py
python scripts/train.py
python scripts/evaluate.py
python -m uvicorn backend.api:app --reload --port 8000
```

In a second terminal, install and start the production-style web frontend:

```powershell
cd web
npm install
npm run dev
```

Copy `web/.env.example` to `web/.env.local` when the API is not running at
`http://localhost:8000`. The web application opens at `http://localhost:3000`.
The command center uses a graceful demo state when the API is offline, and
switches to live/model-backed values when the FastAPI adapter is available.

Always run these commands from the repository root and use the same Python
interpreter for installation, training, evaluation, and the API. The startup
validator prints the executable, Python version, package versions, required
folders, configuration status, and model availability.

The primary frontend is the Next.js command center at `http://localhost:3000`.
It consumes the FastAPI adapter at `http://localhost:8000`.

## Open-world unknown-fault discovery

Flagged observations are first classified by the known-fault model. A label is
accepted only when its confidence exceeds `unknown_discovery.confidence_threshold`.
Low-confidence or unsupported labels become `UNKNOWN`; they are converted into
numeric anomaly signatures and clustered with DBSCAN. Cluster `-1` is isolated
noise and is not promoted to a new fault automatically. Recurring clusters are
written to `data/unknown_faults/registry.json` with occurrence, station, and
time-range metadata.

This is an integration of established anomaly scoring, supervised diagnosis,
distance-based clustering, and persistence tracking. The project-specific
research direction is the explicit open-world rejection and registry loop for
AWS fault signatures. It is not a claim of globally novel algorithms.

All generated data is clearly labeled `SYNTHETIC_NORMAL` or
`SYNTHETIC_FAULT_INJECTED` in the `source` column and is never presented as
real observational data.

## Using real AWS data

Provide a CSV with at least these columns and point the pipeline scripts at it
instead of the generated sample:

```
timestamp, station_id, latitude, longitude, temperature_c, pressure_hpa, humidity_pct
```

No fault labels are required for real-world inference — labels
(`true_state`, `fault_type`) are only used for training/evaluation against
synthetic fault injection, and are never fed to the model as input features.

## Evaluation results (synthetic fault injection)

Running `evaluate.py` on the bundled demo dataset (12,096 readings, 6
stations, 8 injected known-fault types) currently gives approximately:

- Detection precision ≈ 0.30, recall ≈ 0.69, F1 ≈ 0.42 (binary healthy-vs-fault)
- Root-cause classifier weighted F1 ≈ 0.98 (dominated by the "none"/healthy
  class; per-fault-type performance varies — see `train.py` output)

These numbers describe **synthetic fault injection evaluation only** and are
not validated against real-world AWS sensor faults. Detection thresholds in
`config.yaml` (`anomaly.score_threshold`, `vpd.inconsistency_threshold`, etc.)
are tunable and were set for a reasonable demo balance, not tuned against any
production dataset — expect to re-tune once real AWS data is available.

## Design notes

- **No hard-coded thresholds in source** — every threshold lives in
  `config.yaml` (`src/config.py` loads it, with defaults as a fallback).
- **No data leakage**: ground-truth fault labels are never used as classifier
  input features; only for scoring evaluation.
- **CPU-only**: runs on an ordinary laptop (~8 GB RAM), no GPU or heavy deep
  learning frameworks required.
- **Spatial check has no live network dependency**: station coordinates and a
  precomputed Haversine distance matrix are used instead of a live
  OSM/Overpass API call, so the system works fully offline.

## Tests

```bash
pytest tests/
```
