from __future__ import annotations

import logging

import httpx
from fastapi import FastAPI

from shared_contracts.config import get_settings
from shared_contracts.http_clients import HttpAlertClient
from shared_contracts.interfaces import AlertClient, PredictionService
from shared_contracts.models import Alert, FeatureVector, Label, Prediction, Severity

from .service import StubModel, ThresholdPredictionService

log = logging.getLogger("engine")


def create_app(service: PredictionService | None = None, alerts: AlertClient | None = None) -> FastAPI:
    s = get_settings()
    service = service or ThresholdPredictionService(StubModel(), s.anomaly_threshold, s.model_version)
    alerts = alerts or HttpAlertClient(s.gateway_url)
    app = FastAPI(title="engine")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "engine"}

    @app.post("/v1/predict")
    def predict(features: FeatureVector) -> Prediction:
        pred = service.predict(features)
        if pred.label == Label.ANOMALY:
            try:
                alerts.send_alert(Alert(
                    event_id=pred.event_id, score=pred.score,
                    severity=Severity.CRITICAL if pred.score >= 0.95 else Severity.HIGH,
                    message=f"Anomaly score {pred.score:.2f} >= {pred.threshold}"))
            except httpx.HTTPError:
                log.exception("alert delivery failed")  # prediction must still succeed
        return pred

    return app


app = create_app()