import json
import logging
import math
import time
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("suricata-anomaly")

EXCLUDED_FEATURE_COLUMNS = {
    "timestamp",
    "src_ip",
    "dest_ip",
    "flow_id",
    "in_iface",
    "pkt_src",
    "app_proto",
    "proto",
    "anomaly",
    "anomaly_prediction",
    "prediction_label",
    "predicted_label",
}


def parse_suricata_payload(file_bytes: bytes, chunk_size: int = 5000) -> pd.DataFrame:
    """Parse JSON array, single JSON object, or JSONL in bounded chunks."""
    text = file_bytes.decode("utf-8")
    stripped = text.strip()
    if not stripped:
        return pd.DataFrame()

    if stripped[0] in "[{":
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return pd.DataFrame(parsed)
            return pd.DataFrame([parsed])
        except json.JSONDecodeError:
            pass

    frames = []
    batch = []
    for line in text.splitlines():
        if not line.strip():
            continue
        batch.append(json.loads(line))
        if len(batch) >= chunk_size:
            frames.append(pd.DataFrame(batch))
            batch = []
    if batch:
        frames.append(pd.DataFrame(batch))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, copy=False)


def flatten_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in list(df.columns):
        values = df[col].dropna()
        if values.empty:
            continue
        if any(isinstance(value, dict) for value in values):
            mask = df[col].apply(lambda value: isinstance(value, dict))
            flat = pd.json_normalize(df.loc[mask, col]).add_prefix(f"{col}_")
            flat.index = df[mask].index
            df = df.drop(columns=[col]).join(flat)
        elif any(isinstance(value, list) for value in values):
            df[col] = df[col].apply(lambda value: json.dumps(value) if isinstance(value, list) else value)
    return df


def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["object"]).columns:
        nunique = df[col].nunique(dropna=True)
        total = len(df[col])
        if total and 0 < nunique <= min(1000, total * 0.5):
            df[col] = df[col].astype("category")
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in df.select_dtypes(include=np.number).columns
        if col not in EXCLUDED_FEATURE_COLUMNS
    ]
    categorical_cols = [
        col
        for col in df.select_dtypes(include=["object", "bool", "category"]).columns
        if col not in EXCLUDED_FEATURE_COLUMNS
    ]

    numeric = df[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=df.index)
    for col in numeric.columns:
        if numeric[col].isnull().any():
            numeric[col] = numeric[col].fillna(numeric[col].mean())
    if not numeric.empty:
        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0)
        std = numeric.std(axis=0).replace(0, 1)
        numeric = ((numeric - numeric.mean(axis=0)) / std).astype(np.float32)

    encoded = (
        pd.get_dummies(df[categorical_cols], dummy_na=True, drop_first=True, dtype=np.int8)
        if categorical_cols
        else pd.DataFrame(index=df.index)
    )

    features = pd.concat([numeric, encoded], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    if not features.empty:
        features = features.astype(np.float32, copy=False)
    return features


def build_stats(df: pd.DataFrame) -> dict[str, float | int]:
    total = int(len(df))
    anomaly_count = int((df["anomaly"] == -1).sum()) if "anomaly" in df.columns else 0
    normal_count = int((df["anomaly"] == 1).sum()) if "anomaly" in df.columns else 0
    anomaly_percentage = round((anomaly_count / total * 100), 2) if total else 0.0
    normal_percentage = round((normal_count / total * 100), 2) if total else 0.0
    return {
        "total_records": total,
        "normal": normal_count,
        "anomaly": anomaly_count,
        "normal_percentage": normal_percentage,
        "anomaly_percentage": anomaly_percentage,
    }


def dataframe_memory_mb(df: pd.DataFrame) -> float:
    return round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 2)


def process_cpu_percent(process_cpu_seconds: float, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return 0.0
    return round(min(100.0, process_cpu_seconds / elapsed_seconds * 100), 2)


def run_detection_pipeline(file_bytes: bytes, contamination: float = 0.01) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    logger.info("prediction start")

    df = parse_suricata_payload(file_bytes)
    if df.empty:
        raise ValueError("File produced no rows.")

    df = flatten_nested_columns(df)
    df = optimize_dataframe_dtypes(df)
    features = prepare_features(df)
    if features.empty:
        raise ValueError("No usable features.")

    clf = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=1,
    )
    values = features.to_numpy(dtype=np.float32, copy=False)
    clf.fit(values)

    df = df.copy()
    df["anomaly"] = clf.predict(values).astype(np.int8)
    df["anomaly_prediction"] = df["anomaly"]
    df["anomaly_score"] = clf.decision_function(values).astype(np.float32)
    df["prediction_label"] = np.where(df["anomaly"] == -1, "Anomaly", "Normal")
    df["predicted_label"] = np.where(df["anomaly"] == -1, 1, 0).astype(np.int8)
    df = optimize_dataframe_dtypes(df)

    elapsed = time.perf_counter() - started
    cpu_elapsed = time.process_time() - cpu_started
    stats = build_stats(df)
    telemetry = {
        **stats,
        "processing_time_seconds": round(elapsed, 4),
        "cpu_usage_percent": process_cpu_percent(cpu_elapsed, elapsed),
        "memory_usage_mb": dataframe_memory_mb(df) + dataframe_memory_mb(features),
        "feature_columns": int(features.shape[1]),
    }

    logger.info(
        "prediction end",
        extra={
            "processing_time_seconds": telemetry["processing_time_seconds"],
            "cpu_usage_percent": telemetry["cpu_usage_percent"],
            "memory_usage_mb": telemetry["memory_usage_mb"],
            "total_records": telemetry["total_records"],
            "normal": telemetry["normal"],
            "anomaly": telemetry["anomaly"],
        },
    )
    return df, features, telemetry


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    safe_df = df.copy()
    for col in safe_df.select_dtypes(include=["category"]).columns:
        safe_df[col] = safe_df[col].astype(object)
    return [
        {key: _json_safe(value) for key, value in row.items()}
        for row in safe_df.to_dict(orient="records")
    ]


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue().encode("utf-8")
