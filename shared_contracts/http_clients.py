"""Default HTTP implementations of cross-module clients."""
from __future__ import annotations

import httpx

from .interfaces import AlertClient, InferenceClient, ProcessingClient
from .models import Alert, FeatureVector, Prediction, RawEvent


class HttpProcessingClient(ProcessingClient):
    def __init__(self, base_url: str, timeout: float = 5.0):
        self._url, self._timeout = base_url.rstrip("/"), timeout

    def process(self, event: RawEvent) -> Prediction:
        r = httpx.post(f"{self._url}/v1/process", content=event.model_dump_json(),
                       headers={"content-type": "application/json"}, timeout=self._timeout)
        r.raise_for_status()
        return Prediction.model_validate(r.json())


class HttpInferenceClient(InferenceClient):
    def __init__(self, base_url: str, timeout: float = 5.0):
        self._url, self._timeout = base_url.rstrip("/"), timeout

    def predict(self, features: FeatureVector) -> Prediction:
        r = httpx.post(f"{self._url}/v1/predict", content=features.model_dump_json(),
                       headers={"content-type": "application/json"}, timeout=self._timeout)
        r.raise_for_status()
        return Prediction.model_validate(r.json())


class HttpAlertClient(AlertClient):
    def __init__(self, base_url: str, timeout: float = 5.0):
        self._url, self._timeout = base_url.rstrip("/"), timeout

    def send_alert(self, alert: Alert) -> None:
        r = httpx.post(f"{self._url}/internal/alerts", content=alert.model_dump_json(),
                       headers={"content-type": "application/json"}, timeout=self._timeout)
        r.raise_for_status()