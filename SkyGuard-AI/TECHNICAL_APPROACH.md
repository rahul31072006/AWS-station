x# SkyGuard AI Technical Approach

## Architecture

SkyGuard AI uses a layered architecture:

- **Python ML pipeline:** validation, temporal baselines, physics/VPD checks, anomaly scoring, spatial consistency, known-fault classification, explainability, unknown-fault discovery, and self-healing.
- **FastAPI adapter:** exposes pipeline outputs as REST and WebSocket data without rewriting the ML logic.
- **Next.js frontend:** provides routed, responsive operator views over the API.

## Data Flow

```text
AWS CSV / observations
        |
        v
Schema and temporal processing
        |
        v
Baseline + physics + anomaly scoring
        |
        v
Spatial evidence + known-fault classification
        |
        +--> Explainability
        +--> UNKNOWN rejection, signatures, clustering, registry
        +--> Self-healing with original values preserved
        |
        v
FastAPI REST/WebSocket adapter
        |
        v
Next.js command center and routed operator pages
```

## Frontend Approach

The Next.js App Router isolates each operational section:

- `/command-center`
- `/live-stations`
- `/anomaly-detection`
- `/station-intelligence`
- `/ai-explainability`
- `/self-healing`
- `/unknown-discovery`
- `/analytics`
- `/reports`
- `/settings`

The existing dark SkyGuard visual system is shared across routes. Live Stations uses Leaflet with OpenStreetMap tiles, marker status derived from backend data, connected list/map selection, search, filters, station popups, and polling refresh. A separate mock service is used only when the API is unavailable.

## Backend Integration

The FastAPI adapter exposes dashboard, station, anomaly, explanation, healing, unknown-fault, health, and cache-refresh endpoints, plus `/ws/live` for live snapshots. It reads the existing generated data and trained model, so the frontend remains a visualization and operations layer rather than a second ML implementation.

## Open-World Fault Discovery

Flagged observations are accepted as known faults only when classifier confidence meets the configured threshold. Otherwise they become `UNKNOWN`, receive a numeric signature, and are clustered with DBSCAN. Noise is not promoted automatically. Recurring clusters are persisted in the unknown-fault registry.

## Reliability and Validation

- Original sensor values are retained during self-healing.
- SHAP remains optional with a feature-importance fallback.
- The frontend supports API-backed and offline demo states.
- Python tests cover the existing pipeline and discovery/self-healing modules.
- Next.js production builds and TypeScript checks validate the web application.

## Run Commands

```powershell
python -m uvicorn backend.api:app --reload --port 8000
cd web
npm install
npm run dev
```

Open `http://localhost:3000` after both services start.
