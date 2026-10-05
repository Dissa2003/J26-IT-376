import pytest
from fastapi.testclient import TestClient

from processing.app import create_app
from processing.pipeline import NumericFeatureTransformer
from shared_contracts.interfaces import InferenceClient
from shared_contracts.models import Label, Prediction, RawEvent


class FakeInference(InferenceClient):
    def predict(self, features):
        assert features.features == {"x": 1.0}
        return Prediction(event_id=features.event_id, score=0.2, label=Label.NORMAL, threshold=0.8, model_version="t")


def test_transform_keeps_numeric_only():
    fv = NumericFeatureTransformer().transform(RawEvent(source="s", payload={"x": 1, "n": "a"}))
    assert fv.features == {"x": 1.0}


def test_process_ok_and_invalid():
    c = TestClient(create_app(inference=FakeInference()))
    assert c.post("/v1/process", json={"source": "s", "payload": {"x": 1}}).status_code == 200
    assert c.post("/v1/process", json={"source": "", "payload": {}}).status_code == 422