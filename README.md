# LogiShield – Neuro-Symbolic AI Verification for API Gateways

**Research group J26-IT-376** · four research components, one per member, developed in one
repository and connected through `shared_contracts/`.

> **About this branch.** This README describes `integration/member4-dev-sync`: the group `dev`
> branch merged with Member 4's development work, plus Member 4's model-initialization fix and
> runtime adapters. Everything listed as *implemented* below was checked against this branch's
> files. Each component's own documentation remains the authority on that component's details,
> results and limitations.

## Contents

1. [Project overview](#1-project-overview)
2. [Research components and members](#2-research-components-and-members)
3. [System architecture](#3-system-architecture)
4. [Repository structure](#4-repository-structure)
5. [Implemented on this branch](#5-implemented-on-this-branch)
6. [Setup and dependencies](#6-setup-and-dependencies)
7. [Services, ports and demos](#7-services-ports-and-demos)
8. [Tests](#8-tests)
9. [Integration status](#9-integration-status)
10. [Limitations](#10-limitations)
11. [Branches and Git workflow](#11-branches-and-git-workflow)
12. [Component documentation](#12-component-documentation)

## 1. Project overview

An API gateway has to decide, request by request, whether to allow or block traffic. Purely
statistical anomaly detectors are fast but opaque and prone to false positives; purely
rule-based checks are explainable but only cover what the rules anticipate. LogiShield
investigates a **neuro-symbolic** combination of the two:

- API requests are intercepted and structured into a unified, privacy-masked context;
- access and security policies are encoded and checked with formal (SMT-based) verification;
- a lightweight AI model scores the request together with the verified rule information;
- a runtime verification layer combines the formal rules and the AI score into an explainable
  **Allow / Block verdict with an XAI trace**.

The repository is a research prototype. Each component is developed and tested by its owner;
the end-to-end runtime connection between the components is still in progress (see
[§9](#9-integration-status)).

## 2. Research components and members

| Member | Component | Main output |
|---|---|---|
| Member 1 | Data Processing & Context Structuring (Ingestion) | Structured Unified API Context |
| Member 2 | Deterministic Constraint Encoding & Formal Verification (Policy Verification) | Rule Cache, LTL Formula, JSON Vector / Verified Rule Outputs |
| Member 3 | Runtime LTL Verification & Hybrid Execution Layer (Runtime Verification) | Allow/Block Verdict, XAI Trace |
| Member 4 | Hybrid Structure-Aware Tokenization & Lightweight AI Prediction (Tokenization AI) | `AnomalyPrediction(event_id, score, model_version)` |

### Member 1 – Data Processing & Context Structuring
- **Responsibilities:** ingress interception, buffering/serialization, data masking, context dispatching
- **Output:** Structured Unified API Context

### Member 2 – Deterministic Constraint Encoding & Formal Verification
- **Responsibilities:** LTL/SMT rule encoding, neuro-symbolic constraint layer, formal verification engine
- **Output:** Rule Cache, LTL Formula, JSON Vector / Verified Rule Outputs
- **Implemented on this branch:** a deterministic policy grammar, first-order/SMT encoding with
  Z3, conflict detection and a rule cache (`api/`, `src/policy_verifier/`). No temporal (LTL)
  operators or LTL formula output appear in the code; LTL encoding remains part of the proposed
  scope.

### Member 3 – Runtime LTL Verification & Hybrid Execution Layer
- **Responsibilities:** Z3 runtime rule engine, hybrid circuit breaker, mock API gateway router
- **Output:** Allow/Block Verdict, XAI Trace
- **Implemented on this branch:** a Z3-based check of a request's context against a small set of
  safety invariants (class `RuntimeLTLVerifier`, `engine/verifier.py`), a hybrid circuit breaker
  and a mock gateway. Despite the class name, the code checks invariants over a single request
  context; it does not evaluate temporal-logic (LTL) formulas over event sequences, so full
  runtime LTL verification is proposed scope, not demonstrated here.

### Member 4 – Hybrid Structure-Aware Tokenization & Lightweight AI Prediction
- **Responsibilities:** input fusion, privacy-aware masking, normalization, structure-aware
  tokenization, token IDs and tensor preparation, lightweight anomaly scoring
- **Output:** `AnomalyPrediction(event_id, score, model_version)` – an uncalibrated score in
  [0, 1]. It is not an attack probability and not an Allow/Block decision; any threshold and
  the final decision belong to Member 3.

## 3. System architecture

Research pipeline (target architecture):

```
Member 1  Structured Unified API Context ──┐
Member 2  Verified Rule Information ────────┤
                                            ▼
Member 4  Input Fusion
            -> Privacy-Aware Masking
            -> Normalization
            -> Hybrid Structure-Aware Tokenization
            -> Token IDs + Tensor Preparation (fixed length 128)
            -> Hashed n-gram Features + Logistic Regression
            -> AnomalyPrediction(event_id, score, model_version)
                                            ▼
Member 3  Runtime Verification (Z3 rules + hybrid circuit breaker)
            -> Allow / Block Verdict + XAI Trace
```

Design rules for the code base:

- **Intended rule:** component packages should depend on `shared_contracts` rather than on each
  other's internals, so that members can work independently. This is a design convention, not
  something enforced by tooling, and the code on this branch departs from it in these places:
  - `api/routes.py` (Member 2) imports its own `src.policy_verifier` package; that package does
    not use `shared_contracts`.
  - `tokenization_ai` (Member 4) imports only `shared_contracts` from other packages in its
    runtime code (`adapters.py`, `app.py`, `dashboard.py`). Its `runtime_handoff.py` and
    `request_decision_adapter.py` refer to Member 3's and Member 2's interfaces only in
    documentation, not through imports.
  - Cross-component imports occur in tests and benchmarks: `tests/test_e2e_pipeline.py` imports
    `api_gateway`, `ingestion`, `engine` and `tokenization_ai`; `tests/tokenization_ai/test_ingestion_integration.py`
    imports `ingestion`; Member 2's tests import `api` and `src.policy_verifier`; and
    `benchmarks/evaluate_ingestion_ablation.py` imports `ingestion`.
  - Services otherwise communicate over HTTP using URLs from `shared_contracts.config`: the
    gateway calls ingestion, the engine calls the gateway (`/internal/alerts`), and the 8010
    dashboard forwards `/api/v1/*` to the policy service.
- The `api_gateway`, `ingestion`, `engine` and `tokenization_ai` services expose
  `create_app(...)` with injectable dependencies so they can be tested with fakes.
- The arrows above are the research design. On this branch the Member 2 → Member 4 and
  Member 4 → Member 3 links exist as tested adapters but are **not yet wired** into a running
  path; see [§9](#9-integration-status).

## 4. Repository structure

| Path | Owner | Contents |
|---|---|---|
| `ingestion/` | Member 1 | FastAPI ingestion service: interception, validation, AST-based PII masking, broker (Redis or in-memory), context vector builder, Protobuf schema (`ingestion_context.proto`) |
| `benchmarks/` | Member 1 | Ingestion ablation/SLA benchmark, Locust and k6 load scripts |
| `api/` | Member 2 | FastAPI policy verification service (`api.routes`) |
| `src/policy_verifier/` | Member 2 | Policy parser and grammar (`GRAMMAR.md`), logic/calculus, Z3-based verifier, conflict detection, rule cache, request evaluation, evaluation datasets and runners |
| `docs/output_contracts/`, `docs/research/` | Member 2 | `verify-request` response schema; assumption-ablation and ground-truth review notes |
| `engine/` | Member 3 | Runtime engine service: Z3 runtime verifier (`verifier.py`), hybrid circuit breaker, threshold prediction service |
| `api_gateway/` | Member 3 | Mock API gateway: API-key protected event submission, alert list, webhook notifier |
| `tokenization_ai/` | Member 4 | Tokenization AI pipeline, model, HTTP API, `/demo` and `/dashboard`, evaluation and held-out evaluation infrastructure |
| `shared_contracts/` | All | Pydantic DTOs, ABC interfaces, settings, HTTP clients |
| `tests/` | All | Per-component tests and `tests/test_e2e_pipeline.py` (in-process cross-component test) |
| `scripts/` | Group demo | `start_demo.ps1` / `stop_demo.ps1` (policy service + dashboard) |
| `storage/database.py` | – | MongoDB logging helper (`pymongo`); not imported by any service on this branch |
| `docker-compose.yml`, `Dockerfile`, `.env.example` | All | Container build and local configuration |

Note: the repository was created from a generic modular base. Some legacy folders may still
contain placeholder/generic implementations that do not yet fully reflect the approved research
responsibilities. The responsibility definitions in [§2](#2-research-components-and-members)
are the approved project source of truth.

## 5. Implemented on this branch

Summaries below come from the source files on this branch; detailed claims and evaluation
results belong to each component's owner.

**Member 1 – Ingestion (`ingestion/`, port 8001).** Endpoints `GET /health`,
`GET /v1/context-stream`, `POST /v1/intercept`, `POST /v1/ingest`, `POST /v1/threat-score`.
Implements request validation, AST-based PII masking, broker publication (Redis, with in-memory
fallback) and unified context vector building. A GitHub Actions workflow
(`.github/workflows/ingestion_ci.yml`) runs the ingestion tests and the ablation benchmark.

**Member 2 – Policy Verification (`api/` + `src/policy_verifier/`, port 8002).** Endpoints
under `/api/v1/`: `verify-policies`, `verify-text`, `verify-request`, `verification-records`,
`rule-cache`, `conflicts`. Implements policy parsing, logical encoding, Z3-based verification,
conflict detection and a rule cache. The `verify-request` response schema is in
[docs/output_contracts/](docs/output_contracts/README.md).

**Member 3 – Runtime Verification (`engine/` port 8003, `api_gateway/` port 8000).** The
engine exposes `POST /v1/predict`, `POST /v1/verify/runtime` and
`GET /v1/circuit-breaker/status`, combining a Z3 runtime verifier with a hybrid circuit breaker.
Its anomaly score currently comes from `StubModel` (`engine/service.py`). The gateway exposes
`POST /v1/events` (API key required), `GET /v1/alerts` and `POST /internal/alerts`.

**Member 4 – Tokenization AI (`tokenization_ai/`, port 8004).**

| Stage | Implementation |
|---|---|
| Input fusion | `fusion.py` – joins the API context and verified-rule information by `event_id`; a mismatch is rejected |
| Privacy masking | `masking.py` – name- and shape-based masking (`[MASKED_AUTH]`, `[MASKED_SECRET]`, `[MASKED_PII]`, `[MASKED_IP]`) with masked-path provenance |
| Normalization | `normalization.py` – canonical method, path, header names and payload keys |
| Structure-aware tokenization | `tokenizer.py` – fixed grammar of control tags plus literals; every token carries a role (`control`, `literal_api`, `literal_rule`, `masked`) |
| Token IDs and tensorization | `vocabulary.py`, `tensorizer.py` – frozen vocabulary, fixed length 128, attention mask, payload-only truncation that keeps the rule segment |
| Lightweight prediction | `model.py` – hashed unigram/bigram token-ID features + logistic regression (NumPy); refuses to predict untrained |
| Orchestration and API | `service.py`, `app.py` – `POST /v1/analyze` (privacy-safe response), `GET /health`, `/demo`, `/dashboard` |
| Adapters (tested, not wired) | `request_decision_adapter.py` (Member 2 `verify-request` response → rule input), `runtime_handoff.py` (`runtime-verification-input-1` for Member 3's `evaluate_runtime`), `anomaly_signal.py`, `policy_report_adapter.py` |
| Evaluation | `evaluation/` (frozen KPI 1–4 protocols and fixtures), `heldout_eval/` (held-out evaluation infrastructure; no confirmatory result yet) |

Not implemented (and not claimed): learned embeddings, a rule-aligned signal, an anomaly label
or threshold, and a runtime connection to Member 3.

Member 4 evaluation evidence (synthetic, reported in
[tokenization_ai/README.md](tokenization_ai/README.md) §8): KPI 1 input alignment 31/31, KPI 2
token mapping 31/31, KPI 3 tensor correctness 30/30 on synthetic cases; KPI 4 mean in-process
pipeline time 0.3413 ms (historical measurement at commit `ea50a49`, one machine); KPI 5
downstream usability **not measured** (requires Member 3 integration).

## 6. Setup and dependencies

Python 3.12 or later (the Docker image and CI use 3.12).

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r tokenization_ai\requirements-ml.txt   # numpy, needed by tokenization_ai
copy .env.example .env      # for local runs, set the service URLs to http://localhost:<port>
```

On Linux/macOS use `python3 -m venv .venv`, `source .venv/bin/activate` and `cp`.

| File | Contents |
|---|---|
| `requirements.txt` | `fastapi`, `uvicorn[standard]`, `pydantic`, `httpx`, `redis`, `protobuf`, `pytest`, `z3-solver`, `pymongo` |
| `tokenization_ai/requirements-ml.txt` | `numpy>=1.26` – Member 4 only; **not** in `requirements.txt` |

Redis is optional for local development (ingestion falls back to an in-memory broker); Docker
Compose starts one.

## 7. Services, ports and demos

| Port | Service | Start locally | In `docker-compose.yml` |
|---|---|---|---|
| 8000 | API gateway | `python -m api_gateway` | yes (`api_gateway`) |
| 8001 | Ingestion | `python -m ingestion` | yes (`ingestion`) |
| 8002 | Policy verification | `python -m api` | yes (`policy_verification`) |
| 8003 | Runtime engine | `python -m engine` | yes (`engine`) |
| 8004 | Tokenization AI (127.0.0.1 only) | `python -m tokenization_ai` | **no** |
| 8010 | Dashboard with policy pages (127.0.0.1) | `scripts\start_demo.ps1` | **no** |

Compose also starts `redis` and a mock webhook (`mock_webhook`).

**Container stack (Members 1–3):**

```bash
docker compose up --build
curl -X POST localhost:8000/v1/events -H "x-api-key: dev-key-1" -H "content-type: application/json" \
     -d '{"source":"demo","payload":{"temp":0.95}}'
curl localhost:8000/v1/alerts -H "x-api-key: dev-key-1"
```

**Group dashboard demo (Windows, uses `.venv`):**

```powershell
.\scripts\start_demo.ps1      # starts the policy service (8002) and the dashboard (8010), then smoke-tests them
.\scripts\stop_demo.ps1
```

Open `http://127.0.0.1:8010/dashboard#/policy-verification`. The dashboard is served by
`tokenization_ai.app`; its policy pages forward `/api/v1/*` calls to the policy service.

**Member 4 standalone demo:**

```powershell
.venv\Scripts\python.exe -m tokenization_ai         # http://127.0.0.1:8004/demo and /dashboard
.venv\Scripts\python.exe -m tokenization_ai.demo    # console run of the two synthetic fixtures
```

The demo uses synthetic requests, mock rule fixtures and a model fitted on two synthetic
examples; it demonstrates that the stages run end to end, not detection quality.

## 8. Tests

Install both requirement files first ([§6](#6-setup-and-dependencies)); the Member 4 tests and
`tests/test_e2e_pipeline.py` import NumPy. Tests use in-process clients and fakes; no running
services are needed.

```powershell
.venv\Scripts\python.exe -m pytest -q                                   # whole repository
.venv\Scripts\python.exe -m pytest tests/ingestion                      # Member 1
.venv\Scripts\python.exe -m pytest tests/test_policy_verifier.py tests/test_policy_evaluation.py tests/test_policy_request_decision.py tests/test_policy_api_records.py   # Member 2
.venv\Scripts\python.exe -m pytest tests/engine tests/api_gateway       # Member 3
.venv\Scripts\python.exe -m pytest tests/tokenization_ai                # Member 4
.venv\Scripts\python.exe -m pytest tests/test_e2e_pipeline.py           # in-process cross-component test
```

Observed result (30 October 2026, on the pre-rebase integration checkout, Windows, Python 3.14.7): whole repository
**2370 passed, 1 skipped**, of which 2283 tests are collected from `tests/tokenization_ai`.
Passing tests establish behaviour against the tests' own fixtures; they are not evidence of
research KPI attainment or detection quality.

CI: only `.github/workflows/ingestion_ci.yml` exists; it installs `requirements.txt` and runs
the ingestion tests and benchmark. The other components' tests are not run in CI.

## 9. Integration status

| Link | Status on this branch |
|---|---|
| Gateway → Ingestion | Implemented: `POST /v1/events` forwards to ingestion `/v1/ingest` |
| Member 1 → Member 4 | Tested **in-process** with ingestion's real app output (`tests/tokenization_ai/test_ingestion_integration.py`); no networked connection |
| Member 2 → Member 4 | `request_decision_adapter.py` converts Member 2's real `verify-request` response and is tested against Member 2's API in-process; **not wired** – Member 4's demos use mock rule fixtures. The response carries no `event_id`, so request correlation is still to be agreed |
| Member 4 → Member 3 | `runtime_handoff.py` is argument-compatible with the engine's `evaluate_runtime` and tested against it; **not wired**. The engine still scores with `StubModel` |
| Gateway → Member 4 | Not implemented; `tokenization_ai` is not a Compose service |
| Dashboard → Member 2 | HTTP proxy from the 8010 dashboard to the policy service, used by the policy pages; smoke-tested by `start_demo.ps1`. Presentation only, not part of the request pipeline |
| End-to-end test | `tests/test_e2e_pipeline.py` runs gateway, ingestion, Member 4 analysis and Member 3's circuit breaker in one process, but passes a hard-coded score (`0.95`) to the circuit breaker rather than Member 4's prediction |

Remaining work needs the relevant owners: routing gateway traffic to Member 4, adding a Member 4
Compose service, feeding Member 4's score into `evaluate_runtime`, agreeing the score threshold
and failure handling, and an event correlation id in Member 2's response. Details:
[tokenization_ai/INTEGRATION_STATUS.md](tokenization_ai/INTEGRATION_STATUS.md).

## 10. Limitations

- **Prototype, not a validated security product.** No component's results here are claimed as
  real-world detection or production performance.
- **Member 4 model.** The bundled model is fitted on two synthetic examples; its score is
  uncalibrated and not an attack probability. No precision, recall or F1 on real traffic exists,
  and the held-out evaluation has no confirmatory result yet. KPI 1–4 evidence is synthetic and
  from one machine; KPI 5 is unmeasured.
- **Masking** in Member 4 is a heuristic blocklist; unlisted field names, secrets in URL path
  segments and free text are not masked. Do not use real credentials.
- **No complete runtime integration** of Member 2 → Member 4 → Member 3 (see
  [§9](#9-integration-status)); mock rules stand in for verified rules in Member 4's demos.
- **Other components' evaluations** (ingestion benchmarks, policy-verification datasets, runtime
  latency checks) are documented by their owners and were not re-measured for this README.
- `.github/CODEOWNERS` still contains placeholder handles and lists `processing/`, which no longer
  exists on this branch.

## 11. Branches and Git workflow

- `main` – intended to be changed only through pull requests with review and passing CI. This is
  the project's stated policy and **could not be confirmed**: the repository contains no
  branch-protection configuration, and the only CI workflow runs the ingestion tests. Check the
  GitHub branch-protection settings to see what is actually enforced.
- `dev` – group integration branch. This branch's work up to `81fd942` was merged into `dev`
  through PR #16; `dev` has since received further work from other members, so `dev` may contain
  changes that are not on this branch.
- Component branches (observed on `origin`): `feature/ingestion` (ingestion);
  `feature/policy_verification`, `feature/independent-request-check` (policy verification);
  `feature/hybrid-circuit-breaker` (runtime engine); `feature/member4-tokenization-ai`
  (Member 4). Each member changes only their own folder.
- Member 4 branches:
  - `feature/member4-tokenization-ai` – Member 4 development history (predates the merge with `dev`);
  - `integration/member4-dev-sync` – Member 4 merged with `dev` (this branch);
  - `docs/member4-readme-checklist1-20261029` – Member 4 component README for CDAP Checklist 1
    (Draft PR #21 to `dev`).
- Changes to `shared_contracts/` use `contract/<topic>` branches, need approval from all four
  members, and should be additive (new optional fields). Bump `SCHEMA_VERSION` for breaking changes.

## 12. Component documentation

- Member 4 (latest component README, submitted for Checklist 1 on
  `docs/member4-readme-checklist1-20261029`, commit `6f61473`):
  [tokenization_ai/README.md at 6f61473](https://github.com/Dissa2003/J26-IT-376/blob/6f61473/tokenization_ai/README.md).
  The copy of that file on this branch is an earlier version than the linked finalized documentation; the links
  below point to this branch's files.
- Member 4: [tokenization_ai/README.md](tokenization_ai/README.md),
  [tokenization_ai/INTEGRATION_STATUS.md](tokenization_ai/INTEGRATION_STATUS.md),
  [tokenization_ai/CONTRACT_PROPOSAL.md](tokenization_ai/CONTRACT_PROPOSAL.md),
  evaluation protocols in [tokenization_ai/evaluation/](tokenization_ai/evaluation/PROTOCOL.md)
- Member 2: [src/policy_verifier/GRAMMAR.md](src/policy_verifier/GRAMMAR.md),
  [docs/output_contracts/README.md](docs/output_contracts/README.md),
  [docs/research/](docs/research/assumption_ablation.md)
- API collections: [docs/postman_ingestion_collection.json](docs/postman_ingestion_collection.json),
  [docs/postman_engine_collection.json](docs/postman_engine_collection.json)
