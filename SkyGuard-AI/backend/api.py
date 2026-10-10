"""FastAPI read-only adapter for the existing SkyGuard pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/sample/labeled_evaluation_data.csv"
STATIONS_PATH = ROOT / "data/sample/stations.csv"
MODEL_PATH = ROOT / "models/root_cause_classifier.joblib"

app = FastAPI(title="SkyGuard AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: dict[str, Any] = {}


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: _json_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _load() -> dict[str, Any]:
    if "result" in _cache:
        return _cache["result"]
    if not DATA_PATH.exists() or not STATIONS_PATH.exists():
        raise HTTPException(503, "Demo data is unavailable. Run the data generation scripts first.")
    from src.classifier import load_classifier
    from src.config import load_config
    from src.pipeline import run_pipeline

    observations = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    stations = pd.read_csv(STATIONS_PATH)
    model = None
    if MODEL_PATH.exists():
        try:
            model = load_classifier(str(MODEL_PATH))
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            # The pipeline remains useful without optional classifier extras;
            # its response already labels the classification as unavailable.
            import logging
            logging.getLogger("skyguard.api").warning("Classifier unavailable: %s", exc)
    result = run_pipeline(observations, stations, model=model, config=load_config(), explain_top_flagged=250)
    _cache["result"] = result
    _cache["observations"] = observations
    _cache["stations"] = stations
    return result


def _with_vpd(frame: pd.DataFrame) -> pd.DataFrame:
    from src.physics import vapor_pressure_deficit_kpa
    out = frame.copy()
    out["vpd"] = vapor_pressure_deficit_kpa(
        out["temperature_c"].fillna(0).to_numpy(),
        out["humidity_pct"].fillna(0).to_numpy(),
    )
    return out


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "operational", "environment": "synthetic_demo", "api_version": app.version}


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    result = _load()
    features = _with_vpd(result["features"])
    flagged = result["flagged"]
    unknowns = result.get("unknowns", flagged.iloc[0:0])
    healed = result["healed"]
    latest_by_station = features.sort_values("timestamp").groupby("station_id").tail(1)
    station_evidence = []
    for _, latest_row in latest_by_station.iterrows():
        station_id = latest_row["station_id"]
        station_flags = flagged[flagged["station_id"] == station_id].sort_values("timestamp")
        evidence_row = station_flags.iloc[-1] if not station_flags.empty else latest_row
        station_evidence.append({
            "station_id": station_id,
            "timestamp": evidence_row.get("timestamp"),
            "ml_anomaly_probability": evidence_row.get("ml_anomaly_probability", evidence_row.get("anomaly_score", 0.0)),
            "physics_validation": evidence_row.get("physics_validation", "PASSED"),
            "nearby_station_ids": evidence_row.get("nearby_station_ids", []),
            "nearby_station_count": evidence_row.get("nearby_station_count", 0),
            "nearby_disagreement_count": evidence_row.get("nearby_disagreement_count", 0),
            "final_decision": evidence_row.get("final_decision", "NO ACTIVE ANOMALY"),
            "decision_confidence": evidence_row.get("decision_confidence", 0.0),
            "final_fault_type": evidence_row.get("final_fault_type"),
            "health_score": latest_row.get("health_score", 100.0),
            "health_status": latest_row.get("health_status", "STABLE"),
            "health_trend": latest_row.get("health_trend", 0.0),
            "contributors": {
                "ML anomalies": latest_row.get("health_anomaly_penalty", 0.0),
                "Physics violations": latest_row.get("health_physics_penalty", 0.0),
                "Spatial mismatch": latest_row.get("health_spatial_penalty", 0.0),
                "Persistence": latest_row.get("health_persistence_penalty", 0.0),
            },
        })
    degrading = [item for item in station_evidence if item["health_status"] == "DEGRADING"]
    anomaly_records = _records(_with_vpd(flagged.sort_values("timestamp", ascending=False).head(100)))
    for record, row_id in zip(anomaly_records, flagged.sort_values("timestamp", ascending=False).head(100).index):
        record["row_id"] = int(row_id)
    return {
        "source": "SYNTHETIC_DEMO",
        "updated_at": features["timestamp"].max(),
        "kpis": {
            "network_health": round(float(features["health_score"].mean()), 1),
            "active_stations": int(features["station_id"].nunique()),
            "anomalies_detected": int(features["is_flagged"].sum()),
            "sensor_faults": int(flagged["is_sensor_fault"].sum()),
            "self_healed": int(healed["imputed"].sum()),
            "unknown_patterns": int(len(unknowns)),
            "sensor_health": round(float(features.groupby("station_id")["health_score"].last().mean()), 1),
            "degrading_sensors": len(degrading),
            "early_warnings": len(degrading),
        },
        "stations": _records(_cache["stations"]),
        "latest": _records(latest_by_station),
        "station_evidence": [_json_value(item) for item in station_evidence],
        "anomalies": anomaly_records,
        "series": _records(features.sort_values("timestamp").groupby("station_id").tail(72)),
        "unknowns": _records(_with_vpd(unknowns)),
        "registry": result.get("unknown_registry", {}),
        "healing": _records(_with_vpd(healed[healed["imputed"]].sort_values("timestamp", ascending=False).head(100))),
        "explanations": [
            {"row_id": int(index), **explanation}
            for index, explanation in result.get("explanations", {}).items()
        ],
    }


@app.get("/api/stations")
def stations() -> list[dict[str, Any]]:
    _load()
    return _records(_cache["stations"])


@app.get("/api/stations/{station_id}")
def station(station_id: str) -> dict[str, Any]:
    data = dashboard()
    latest = next((item for item in data["latest"] if item["station_id"] == station_id), None)
    if latest is None:
        raise HTTPException(404, "Station not found")
    latest["station"] = next(item for item in data["stations"] if item["station_id"] == station_id)
    latest["history"] = [item for item in data["series"] if item["station_id"] == station_id]
    latest["evidence"] = next(item for item in data["station_evidence"] if item["station_id"] == station_id)
    latest["health_evolution"] = [
        {
            "timestamp": item["timestamp"],
            "health_score": item.get("health_score"),
            "health_status": item.get("health_status"),
        }
        for item in latest["history"]
    ]
    return latest


@app.get("/api/anomalies")
def anomalies(limit: int = Query(100, ge=1, le=1000)) -> list[dict[str, Any]]:
    return dashboard()["anomalies"][:limit]


@app.get("/api/unknown-faults")
def unknown_faults() -> dict[str, Any]:
    data = dashboard()
    return {"items": data["unknowns"], "registry": data["registry"]}


@app.get("/api/explanations/{row_id}")
def explanation(row_id: int) -> dict[str, Any]:
    result = _load()
    explanation_data = result.get("explanations", {}).get(row_id)
    if explanation_data is None:
        raise HTTPException(404, "Explanation is unavailable for this row")
    return explanation_data


@app.post("/api/cache/refresh")
def refresh() -> dict[str, str]:
    _cache.clear()
    return {"status": "refreshed"}


@app.websocket("/ws/live")
async def live_updates(websocket: WebSocket) -> None:
    """Push the latest dashboard snapshot to clients when requested."""
    await websocket.accept()
    try:
        await websocket.send_json(dashboard())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        return
