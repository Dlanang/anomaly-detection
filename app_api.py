import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from core import (
    build_stats,
    dataframe_to_csv_bytes,
    dataframe_to_records,
    run_detection_pipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("suricata-anomaly")

api = FastAPI(title="Suricata Anomaly Detection API", version="1.0.0")

_LATEST_DF = None
_LATEST_STATS: dict[str, Any] | None = None


@api.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@api.post("/detect")
async def detect(
    request: Request,
    contamination: float = Query(default=0.01, ge=0.001, le=0.5),
) -> JSONResponse:
    """Run prediction and return every predicted record, including normal rows."""
    global _LATEST_DF, _LATEST_STATS
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="Request body is empty.")

    try:
        df, _features, telemetry = run_detection_pipeline(payload, contamination=contamination)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    _LATEST_DF = df
    _LATEST_STATS = telemetry
    return JSONResponse(
        {
            "stats": telemetry,
            "data": dataframe_to_records(df),
        }
    )


@api.get("/stats")
def stats() -> dict[str, Any]:
    if _LATEST_DF is None:
        return {
            "total_records": 0,
            "normal": 0,
            "anomaly": 0,
            "normal_percentage": 0.0,
            "anomaly_percentage": 0.0,
        }
    return build_stats(_LATEST_DF)


@api.get("/results/json")
def results_json() -> JSONResponse:
    if _LATEST_DF is None:
        raise HTTPException(status_code=404, detail="No prediction result is available. Run POST /detect first.")
    return JSONResponse(
        {
            "stats": build_stats(_LATEST_DF),
            "data": dataframe_to_records(_LATEST_DF),
        }
    )


@api.get("/results/csv")
def results_csv() -> Response:
    if _LATEST_DF is None:
        raise HTTPException(status_code=404, detail="No prediction result is available. Run POST /detect first.")
    return Response(
        content=dataframe_to_csv_bytes(_LATEST_DF),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=suricata_predictions.csv"},
    )
