# LogiShield – Neuro-Symbolic AI Verification for API Gateways

## Pipeline

```
Unified API Context + Verified Rule Information
  -> LogiShield Input Fusion
  -> Privacy-Aware Masking
  -> Normalization
  -> Hybrid Structure-Aware Tokenization
  -> Token / Embedding Mapping
  -> Tensor Preparation
  -> Lightweight AI Prediction
  -> Runtime Verification Runtime Verification
  -> Allow / Block Verdict + XAI Trace
```

## Member responsibilities

### Ingestion – Data Processing & Context Structuring Stage
- **Responsibilities:** Ingress interception, buffering/serialization, data masking, context dispatching
- **Output:** Structured Unified API Context

### Formal Verification – Deterministic Constraint Encoding & Formal Verification
- **Responsibilities:** LTL/SMT rule encoding, neuro-symbolic constraint layer, formal verification engine
- **Output:** Rule Cache, LTL Formula, JSON Vector / Verified Rule Outputs

### LogiShield – Hybrid Structure-Aware Tokenization & Lightweight AI Prediction
- **Responsibilities:** Input fusion, privacy-aware masking, structure-aware tokenization, token/embedding mapping, tensor preparation, lightweight AI prediction
- **Output:** Anomaly Prediction, AI-ready Representation, Rule-aligned Signal, Logs/Trace

### Runtime Verification – Runtime LTL Verification & Hybrid Execution Layer
- **Responsibilities:** Z3 runtime rule engine, hybrid circuit breaker, mock API gateway router
- **Output:** Allow/Block Verdict, XAI Trace

## Target research module map

| Module | Owner | Responsibility |
|---|---|---|
| `ingestion/` | Ingestion | Ingress interception, buffering/serialization, data masking, context dispatching |
| `api/` + `src/policy_verifier/` | Formal Verification | Policy parsing, FOL/SMT verification, conflict detection, and rule cache |
| `tokenization_ai/` | LogiShield | Input fusion, privacy-aware masking, structure-aware tokenization, token/embedding mapping, tensor preparation, lightweight AI prediction |
| `engine/`, `api_gateway/` | Runtime Verification | Z3 runtime rule engine, hybrid circuit breaker, mock API gateway router |
| `shared_contracts/` | All | Pydantic DTOs, ABC interfaces, config, HTTP clients |

Note: The repository was initially created from a generic modular base. Some legacy folders may still contain placeholder/generic implementations that do not yet fully reflect the approved research responsibilities. The responsibility definitions above are the approved project source of truth. This branch implements LogiShield in `tokenization_ai/`.

Modules import **only** `shared_contracts` — never each other. Each service module exposes `create_app(...)` with injectable dependencies so it can be tested with fakes.

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

## Run the demo

```powershell
.\scripts\start_demo.ps1
.\scripts\stop_demo.ps1
```

Open `http://127.0.0.1:8010/dashboard#/policy-verification` after starting.

## Branching
- `main`: protected, PR + 1 review + green CI. Production-ready.
- `develop`: integration branch.
- Module branches: `feature/ingestion-*`, `feature/policy-*`, `feature/engine-*`, `feature/gateway-*`. Touch only your own folder.
- Changes to `shared_contracts/` use `contract/<topic>` branches, need approval from all 4 members (CODEOWNERS), and should be additive (new optional fields) to avoid breaking others. Bump `SCHEMA_VERSION` for breaking changes.
- LogiShield development is currently done on `feature/logishield-tokenization-ai`.

## Per-member guide
Run a single module locally: `python -m <module>` (set URLs in `.env`), test: `pytest tests/<module>`.

1. **Ingestion – ingestion**: ingress interception, buffering/serialization, data masking, and context dispatching, producing the Structured Unified API Context. Test: `pytest tests/ingestion`.
2. **Policy Verification – api + src/policy_verifier**: policy parsing, FOL/SMT verification, conflict detection, and rule cache. Run: `python -m api`; test: `pytest tests/test_policy_verifier.py tests/test_policy_evaluation.py`.
3. **Runtime Verification – engine / api_gateway**: the Z3 runtime rule engine, hybrid circuit breaker, and mock API gateway router, producing the Allow/Block Verdict and XAI Trace. Test: `pytest tests/engine`, `pytest tests/api_gateway`.
4. **LogiShield – tokenization_ai**: input fusion, privacy-aware masking, structure-aware tokenization, token/embedding mapping, tensor preparation, and lightweight AI prediction, producing the Anomaly Prediction, AI-ready Representation, Rule-aligned Signal, and Logs/Trace. Test: `pytest tests/tokenization_ai`.

## Contracts
Research handoffs between members:
- **Ingestion → Policy Verification/LogiShield:** Structured Unified API Context
- **Formal Verification → LogiShield:** Verified Rule Information / JSON Vector
- **LogiShield → Runtime Verification:** Anomaly Prediction, AI-ready Representation, Rule-aligned Signal, Logs/Trace
- **Runtime Verification → caller:** Allow/Block Verdict, XAI Trace
