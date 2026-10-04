# Modular Event-Scoring Platform

Pipeline: `client -> api_gateway -> ingestion -> processing -> engine -> (alert) -> api_gateway`

| Module | Owner | Port | Responsibility |
|---|---|---|---|
| `ingestion/` | Member 1 | 8001 | Event intake, broker (producer/consumer interfaces) |
| `processing/` | Member 2 | 8002 | Schema validation, feature engineering |
| `engine/` | Member 3 | 8003 | Model inference, thresholding, alert creation |
| `api_gateway/` | Member 4 | 8000 | Public REST API, auth/logging/error middleware, notifiers |
| `shared_contracts/` | All | - | Pydantic DTOs, ABC interfaces, config, HTTP clients |

Modules import **only** `shared_contracts` — never each other. Each exposes `create_app(...)` with injectable dependencies so it can be tested with fakes.

## Quick start
```bash
python -m venv .venv && .venv\Scripts\activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # local runs: set service URLs to http://localhost:<port>
pytest                      # all tests (no network needed)
docker compose up --build   # full stack
curl -X POST localhost:8000/v1/events -H "x-api-key: dev-key-1" -H "content-type: application/json" \
     -d '{"source":"demo","payload":{"temp":0.95}}'
curl localhost:8000/v1/alerts -H "x-api-key: dev-key-1"
```

## Branching
- `main`: protected, PR + 1 review + green CI. Production-ready.
- `develop`: integration branch.
- Module branches: `feature/ingestion-*`, `feature/processing-*`, `feature/engine-*`, `feature/gateway-*`. Touch only your own folder.
- Changes to `shared_contracts/` use `contract/<topic>` branches, need approval from all 4 members (CODEOWNERS), and should be additive (new optional fields) to avoid breaking others. Bump `SCHEMA_VERSION` for breaking changes.

## Per-member guide
Run a single module locally: `python -m <module>` (set URLs in `.env`), test: `pytest tests/<module>`.

1. **Member 1 – ingestion**: implement real `EventProducer`/`EventConsumer` (Redis Streams/Kafka; `REDIS_URL`) in `ingestion/broker.py`; add API-proxy/stream sources. Test: `pytest tests/ingestion`.
2. **Member 2 – processing**: extend `BasicValidator` / `NumericFeatureTransformer` in `processing/pipeline.py` (implement `SchemaValidator`, `FeatureTransformer`). Output must be a `FeatureVector`. Test: `pytest tests/processing`.
3. **Member 3 – engine**: replace `StubModel` (`InferenceModel`) in `engine/service.py`; tune thresholds via `ANOMALY_THRESHOLD`. Add gRPC alongside REST if desired. Test: `pytest tests/engine`.
4. **Member 4 – api_gateway**: add notifiers (`Notifier`) in `api_gateway/notifiers.py`, extend middleware/auth in `middleware.py`; `WEBHOOK_URL` points to a mock echo service. Test: `pytest tests/api_gateway`.

## Contracts
`RawEvent` (ingestion→processing), `FeatureVector` (processing→engine), `Prediction` (engine→back up the chain), `Alert` (engine→gateway). Defined in `shared_contracts/models.py`.