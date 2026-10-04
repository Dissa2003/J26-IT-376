from __future__ import annotations

from fastapi import FastAPI, HTTPException

from shared_contracts.config import get_settings
from shared_contracts.http_clients import HttpInferenceClient
from shared_contracts.interfaces import FeatureTransformer, InferenceClient, SchemaValidator
from shared_contracts.models import Prediction, RawEvent

from .pipeline import BasicValidator, NumericFeatureTransformer


def create_app(
    validator: SchemaValidator | None = None,
    transformer: FeatureTransformer | None = None,
    inference: InferenceClient | None = None,
) -> FastAPI:
    settings = get_settings()
    validator = validator or BasicValidator()
    transformer = transformer or NumericFeatureTransformer()
    inference = inference or HttpInferenceClient(settings.engine_url)
    app = FastAPI(title="processing")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "processing"}

    @app.post("/v1/process")
    def process(event: RawEvent) -> Prediction:
        try:
            validator.validate(event)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return inference.predict(transformer.transform(event))

    return app


app = create_app()