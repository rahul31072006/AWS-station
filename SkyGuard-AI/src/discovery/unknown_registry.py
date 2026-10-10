"""Persistent, file-backed registry of recurring unknown-fault patterns."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class UnknownFaultRegistry:
    """Store cluster occurrence summaries without requiring a database."""

    def __init__(self, path: str | Path = "data/unknown_faults/registry.json"):
        self.path = Path(path)
        self.records = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def update(self, clustered: pd.DataFrame) -> dict:
        """Record non-noise clusters and return the complete registry."""
        for cluster_id, group in clustered.groupby("unknown_cluster_id"):
            if int(cluster_id) < 0:
                continue
            key = str(int(cluster_id))
            timestamps = pd.to_datetime(group.get("timestamp"), errors="coerce").dropna()
            stations = sorted(group.get("station_id", pd.Series(dtype=str)).dropna().unique().tolist())
            record = self.records.setdefault(key, {
                "fault_id": f"UF-{int(cluster_id) + 1:03d}",
                "occurrences": 0,
                "stations": [],
                "status": "Emerging",
            })
            record["occurrences"] += int(len(group))
            record["stations"] = sorted(set(record["stations"]) | set(stations))
            if len(timestamps):
                record["first_detected"] = min(record.get("first_detected", timestamps.min().isoformat()), timestamps.min().isoformat())
                record["last_detected"] = max(record.get("last_detected", timestamps.max().isoformat()), timestamps.max().isoformat())
            if record["occurrences"] >= 10:
                record["status"] = "Persistent"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")
        return self.records