# LogiShield – Neuro-Symbolic AI Verification for API Gateways

**Research group J26-IT-376** · four research components, one per member, developed in one
repository and connected through `shared_contracts/`.

> **About this branch.** This README describes `feature/member4-tokenization-ai`, Member 4's
> development branch. It **predates** Member 4's merge with the group `dev` branch: the
> Member 1–3 folders here are the earlier modular base, not the current implementations of
> those members. For the current group code base see `dev` and `integration/member4-dev-sync`.
> Everything listed as *implemented* below was checked against this branch's files.

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

The repository is a research prototype. On this branch only Member 4's component is
implemented at research level; see [§5](#5-implemented-on-this-branch).

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

### Member 3 – Runtime LTL Verification & Hybrid Execution Layer
- **Responsibilities:** Z3 runtime rule engine, hybrid circuit breaker, mock API gateway router
- **Output:** Allow/Block Verdict, XAI Trace

### Member 4 – Hybrid Structure-Aware Tokenization & Lightweight AI Prediction
- **Responsibilities:** input fusion, privacy-aware masking, normalization, structure-aware
  tokenization, token IDs and tensor preparation, lightweight anomaly scoring
- **Output:** `AnomalyPrediction(event_id, score, model_version)` – an uncalibrated score in
  [0, 1]. It is not an attack probability and not an Allow/Block decision; any threshold and
  the final decision belong to Member 3.

These are the approved research responsibilities. On this branch, Members 1–3 are represented
only by the earlier base modules described in [§5](#5-implemented-on-this-branch).

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
Member 3  Runtime Verification
            -> Allow / Block Verdict + XAI Trace
```

Design rules for the code base:

- **Intended convention:** component packages depend on `shared_contracts` rather than on each
  other's internals. This is a design convention, not enforced by tooling. Observed on this
  branch:
  - *Cross-component imports:* none in runtime code. `ingestion`, `processing`, `engine` and
    `api_gateway` import only `shared_contracts` and their own package; `tokenization_ai` imports
    only `shared_contracts` (`adapters.py`, `app.py`, `dashboard.py`) and its own modules.
  - *Internal imports:* each package uses relative imports within itself.
  - *HTTP communication:* services call each other over HTTP with URLs from
    `shared_contracts.config`: gateway → ingestion → processing → engine, and engine → gateway
    (`/internal/alerts`). `tokenization_ai` makes no calls to the other services.
  - *Test imports:* tests import their own module; `tests/tokenization_ai/test_member1_integration.py`
    also imports `ingestion` to exercise Member 4 against ingestion's output in-process.
- Each of the five service packages (`api_gateway`, `ingestion`, `processing`, `engine`,
  `tokenization_ai`) defines `create_app(...)` with optional injectable parameters, so it can be
  tested with fakes. The injectable parameters differ per service (for example a notifier, a
  processing or alert client, or a `TokenizationAIService`), and each module also builds a
  default `app = create_app()` at import time using real settings or, for `tokenization_ai`, the
  synthetic demo service.
- On this branch the arrows into and out of Member 4 are **not connected**: Member 4 runs as a
  standalone service with mock rule fixtures; see [§9](#9-integration-status).

## 4. Repository structure

| Path | Owner | Contents on this branch |
|---|---|---|
| `ingestion/` | Member 1 | Base ingestion service: validation, PII masker, broker (Redis or in-memory), context vector builder; forwards events to `processing` |
| `processing/` | Member 2 (placeholder) | Generic base module: schema validation and a numeric feature transformer that calls the engine. It does **not** implement LTL/SMT encoding or formal verification |
| `engine/` | Member 3 | Base engine: `StubModel` plus a threshold prediction service that raises alerts. No Z3 verifier or circuit breaker on this branch |
| `api_gateway/` | Member 3 | Mock API gateway: API-key protected event submission, alert list, webhook notifier |
| `tokenization_ai/` | Member 4 | Tokenization AI pipeline, model, HTTP API, `/demo` and `/dashboard`, evaluation and held-out evaluation infrastructure |
| `shared_contracts/` | All | Pydantic DTOs, ABC interfaces, settings, HTTP clients |
| `tests/` | All | Per-module tests (`ingestion`, `processing`, `engine`, `api_gateway`, `tokenization_ai`) |
| `docker-compose.yml`, `Dockerfile`, `.env.example` | All | Container build and local configuration |

Note: the repository was created from a generic modular base. Some legacy folders may still
contain placeholder/generic implementations that do not yet fully reflect the approved research
responsibilities. The responsibility definitions in [§2](#2-research-components-and-members)
are the approved project source of truth. On `dev`, Member 2's implementation has since moved to
`api/` + `src/policy_verifier/`, and Member 3's engine has gained a Z3 verifier and hybrid
circuit breaker; neither is on this branch.

## 5. Implemented on this branch

**Members 1–3 (base modules).** A working base request path exists:
gateway `POST /v1/events` (port 8000) → ingestion `POST /v1/ingest` (8001) → processing
`POST /v1/process` (8002) → engine `POST /v1/predict` (8003). The engine's `StubModel` uses the
largest numeric feature value as the score, applies `ANOMALY_THRESHOLD` and posts alerts back to
the gateway. This is the original scaffold, not the members' research implementations.

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
| Standalone modules (tested, not wired) | `anomaly_signal.py` (proposed `AnomalySignal` output format), `policy_report_adapter.py` (caller-supplied Member 2-style policy reports → rule input) |
| Evaluation | `evaluation/` (frozen KPI 1–4 protocols and fixtures), `heldout_eval/` (held-out evaluation infrastructure; no confirmatory result yet) |

Not implemented (and not claimed): learned embeddings, a rule-aligned signal, an anomaly label
or threshold, and any handoff to Member 3 (`adapters.to_member3_output` raises
`NotImplementedError` on this branch).

Member 4 evaluation evidence (synthetic, reported in
[tokenization_ai/README.md](tokenization_ai/README.md) §8): KPI 1 input alignment 31/31, KPI 2
token mapping 31/31, KPI 3 tensor correctness 30/30 on synthetic cases; KPI 4 mean in-process
pipeline time 0.3413 ms (historical measurement at commit `ea50a49`, one machine); KPI 5
downstream usability **not measured** (requires Member 3 integration).

## 6. Setup and dependencies

Python 3.12 or later (the Docker image uses 3.12).

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r tokenization_ai\requirements-ml.txt   # numpy, needed by tokenization_ai
copy .env.example .env      # for local runs, set the service URLs to http://localhost:<port>
```

On Linux/macOS use `python3 -m venv .venv`, `source .venv/bin/activate` and `cp`.

| File | Contents |
|---|---|
| `requirements.txt` | `fastapi`, `uvicorn[standard]`, `pydantic`, `httpx`, `redis`, `protobuf`, `pytest` |
| `tokenization_ai/requirements-ml.txt` | `numpy>=1.26` – Member 4 only; **not** in `requirements.txt` |

This branch has no `z3-solver` or `pymongo` dependency (they arrive with the newer Member 2/3
code on `dev`).

## 7. Services, ports and demos

| Port | Service | Start locally | In `docker-compose.yml` |
|---|---|---|---|
| 8000 | API gateway | `python -m api_gateway` | yes (`api_gateway`) |
| 8001 | Ingestion | `python -m ingestion` | yes (`ingestion`) |
| 8002 | Processing (base module) | `python -m processing` | yes (`processing`) |
| 8003 | Engine (base module) | `python -m engine` | yes (`engine`) |
| 8004 | Tokenization AI (127.0.0.1 only) | `python -m tokenization_ai` | **no** |

Compose also starts `redis` and a mock webhook (`mock_webhook`).

**Container stack (base modules):**

```bash
docker compose up --build
curl -X POST localhost:8000/v1/events -H "x-api-key: dev-key-1" -H "content-type: application/json" \
     -d '{"source":"demo","payload":{"temp":0.95}}'
curl localhost:8000/v1/alerts -H "x-api-key: dev-key-1"
```

**Member 4 demo:**

```powershell
.venv\Scripts\python.exe -m tokenization_ai         # http://127.0.0.1:8004/demo and /dashboard
.venv\Scripts\python.exe -m tokenization_ai.demo    # console run of the two synthetic fixtures
```

The demo uses synthetic requests, mock rule fixtures and a model fitted on two synthetic
examples; it demonstrates that the stages run end to end, not detection quality. This branch has
no group demo script and no policy-verification dashboard pages.

## 8. Tests

Install both requirement files first ([§6](#6-setup-and-dependencies)); the Member 4 tests
import NumPy. Tests use in-process clients and fakes; no running services are needed.

```powershell
.venv\Scripts\python.exe -m pytest -q                                   # whole repository
.venv\Scripts\python.exe -m pytest tests/ingestion tests/processing tests/engine tests/api_gateway   # base modules
.venv\Scripts\python.exe -m pytest tests/tokenization_ai                # Member 4
```

Observed result (30 October 2026, commit `87365a2`, Windows, Python 3.14.7): whole repository
**2244 passed, 1 skipped**, of which 2219 tests are collected from `tests/tokenization_ai`.
Passing tests establish behaviour against the tests' own fixtures; they are not evidence of
research KPI attainment or detection quality.

This branch has no CI workflow and no cross-component end-to-end test.

## 9. Integration status

| Link | Status on this branch |
|---|---|
| Base path gateway → ingestion → processing → engine | Implemented with the scaffold modules (`StubModel` scoring) |
| Member 1 → Member 4 | Tested **in-process** with the ingestion module's output (`tests/tokenization_ai/test_member1_integration.py`); no networked connection |
| Member 2 → Member 4 | **Not integrated.** `policy_report_adapter.py` converts caller-supplied report mappings only; no real Member 2 output is consumed. Member 4's demos use mock rule fixtures |
| Member 4 → Member 3 | **Not integrated.** `anomaly_signal.py` proposes an output format; no handoff exists (`to_member3_output` is `NotImplementedError`) and the engine scores with `StubModel` |
| Member 4 in Compose / gateway | Not configured |

The later runtime handoff and Member 2 `verify-request` adapter exist only on
`integration/member4-dev-sync`. Details for this branch:
[tokenization_ai/INTEGRATION_STATUS.md](tokenization_ai/INTEGRATION_STATUS.md).

## 10. Limitations

- **Prototype, not a validated security product.** No result here is claimed as real-world
  detection or production performance.
- **Members 1–3 on this branch are scaffold code.** Their research implementations (formal
  policy verification, Z3 runtime verification, hybrid circuit breaker, newer ingestion) are on
  `dev`, not here.
- **Member 4 model.** The bundled model is fitted on two synthetic examples; its score is
  uncalibrated and not an attack probability. No precision, recall or F1 on real traffic exists,
  and the held-out evaluation has no confirmatory result yet. KPI 1–4 evidence is synthetic and
  from one machine; KPI 5 is unmeasured.
- **Masking** in Member 4 is a heuristic blocklist; unlisted field names, secrets in URL path
  segments and free text are not masked. Do not use real credentials.
- **No Member 2 → Member 4 → Member 3 integration** on this branch.
- `.github/CODEOWNERS` still contains placeholder handles.

## 11. Branches and Git workflow

- `main` – **intended team workflow:** changes only through pull requests with one review and
  passing CI. This is not verified GitHub enforcement: the repository contains no
  branch-protection configuration, this branch has no CI workflow, and the actual protection
  settings were not checked. Confirm them in the GitHub repository settings.
- `dev` – group integration branch.
- Module branches touch only their own folder. Changes to `shared_contracts/` use
  `contract/<topic>` branches, need approval from all four members, and should be additive (new
  optional fields). Bump `SCHEMA_VERSION` for breaking changes.
- Member 4 branches:
  - `feature/member4-tokenization-ai` – Member 4 development history (this branch);
  - `integration/member4-dev-sync` – Member 4 merged with `dev`, plus runtime adapters;
  - `docs/member4-readme-checklist1-20261029` – Member 4 component README for CDAP Checklist 1
    (Draft PR #21 to `dev`).

## 12. Component documentation

- Member 4 (this branch): [tokenization_ai/README.md](tokenization_ai/README.md). It is the
  historical component README of this branch; some of its figures predate the current test
  suite (for example, it records 871 Member 4 tests as "current", whereas
  [§8](#8-tests) of this README reports the count observed on this branch tip).
- Member 4 (newer, **different branch state**): finalized Checklist 1 documentation at
  [tokenization_ai/README.md at 6f61473](https://github.com/Dissa2003/J26-IT-376/blob/6f61473/tokenization_ai/README.md)
  (`docs/member4-readme-checklist1-20261029`). It describes the later integration/docs state, so
  it must not be read as proof that the capabilities it mentions (such as the runtime handoff and
  `verify-request` adapters) exist on this historical feature branch.
- Member 4 (other files on this branch):
  [tokenization_ai/INTEGRATION_STATUS.md](tokenization_ai/INTEGRATION_STATUS.md),
  [tokenization_ai/CONTRACT_PROPOSAL.md](tokenization_ai/CONTRACT_PROPOSAL.md),
  evaluation protocols in [tokenization_ai/evaluation/](tokenization_ai/evaluation/PROTOCOL.md)
  and [tokenization_ai/heldout_eval/](tokenization_ai/heldout_eval/PROTOCOL.md)
- API collection: [docs/postman_ingestion_collection.json](docs/postman_ingestion_collection.json)
