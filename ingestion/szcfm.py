"""SZ-CFM pipeline: correlate HTTP requests with asynchronous AI assessments.

Stage order (one request):

A. ingest      - size limit, JSON parse, schema validation, server-side ids/timestamp
B. mask        - recursive PII/credential masking (raw authorization is dropped)
D. features    - fixed-order v1 vector (computed before C because the envelope carries it)
E. slot        - allocate a generation-stamped request-state slot
C. serialize   - protobuf envelope, written once into the slot's pre-allocated memory
H1. rules      - authentication, authorization and mandatory rules (no model input)
F. inference   - submitted asynchronously unless a deterministic BLOCK already applies
G. fusion      - result matched by (slot, generation, request_id, correlation_id)
H2. decision   - model or availability policy added; committed exactly once
   forward / publish, then the slot is wiped and released.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from shared_contracts.interfaces import EventProducer
from shared_contracts.models import RawEvent

from .feature_schema import FEATURE_SCHEMA_VERSION, FeatureExtractor
from .inference import (
    AsyncInferenceExecutor,
    DevelopmentHeuristicBackend,
    InferenceBackend,
    InferenceResult,
    InferenceStatus,
    InferenceTask,
)
from .pii_masker import PIIMasker
from .request_state import DeliveryOutcome, RequestStateRing, SlotState, SlotTicket
from .szcfm_policy import FUSION_FORMULA, PolicyEngine, SZCFMConfig, context_risk, fuse, threat_level
from .szcfm_proto import ENVELOPE_MESSAGE_TYPE, ENVELOPE_SCHEMA_VERSION, SZCFMEnvelope
from .szcfm_schema import (
    AIAssessmentResult,
    ContextFusionResult,
    FeatureExtractionResult,
    ForwardingResult,
    PipelineMetrics,
    RequestStateResult,
    SecurityDecisionResult,
    SerializationResult,
    SZCFMEvent,
    SZCFMResult,
)

log = logging.getLogger("ingestion.szcfm")

Forwarder = Callable[[SZCFMEvent, str, str], Any]
EnvelopeSink = Callable[[memoryview, SlotTicket], None]


class PayloadTooLargeError(ValueError):
    pass


class MalformedRequestError(ValueError):
    pass


class EventValidationError(ValueError):
    def __init__(self, errors: list[dict[str, Any]]) -> None:
        super().__init__("event failed schema validation")
        self.errors = errors


class CorrelationConflictError(ValueError):
    pass


def _reject_constant(name: str) -> None:
    raise MalformedRequestError(f"non-finite JSON number {name} is not allowed")


def _safe_errors(exc: ValidationError) -> list[dict[str, Any]]:
    """Validation errors without the offending input values (they may hold secrets)."""
    return [
        {"loc": [str(p) for p in err["loc"]], "msg": err["msg"], "type": err["type"]}
        for err in exc.errors(include_url=False, include_input=False, include_context=False)
    ]


def _ms(ns: int | None) -> float | None:
    return None if ns is None else round(ns / 1_000_000, 4)


class SZCFMMetrics:
    """Thread-safe counters and recent latency samples."""

    _STAGES = ("total", "serialization", "feature_extraction", "ai_inference", "ai_wait", "context_fusion")

    def __init__(self, window: int = 2048) -> None:
        self._lock = threading.Lock()
        self._counters: Counter[str] = Counter()
        self._samples = {stage: deque(maxlen=window) for stage in self._STAGES}
        self._started = time.monotonic()

    def count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[name] += n

    def sample(self, stage: str, ns: int | None) -> None:
        if ns is not None:
            with self._lock:
                self._samples[stage].append(ns)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            samples = {k: sorted(v) for k, v in self._samples.items()}
        uptime = time.monotonic() - self._started
        latency = {}
        for stage, values in samples.items():
            if values:
                latency[stage] = {
                    "count": len(values),
                    "p50_ms": round(values[len(values) // 2] / 1e6, 4),
                    "p95_ms": round(values[min(len(values) - 1, int(len(values) * 0.95))] / 1e6, 4),
                    "max_ms": round(values[-1] / 1e6, 4),
                }
        processed = counters.get("requests_processed", 0)
        return {
            "uptime_s": round(uptime, 3),
            "counters": counters,
            "throughput_rps": round(processed / uptime, 3) if uptime > 0 else 0.0,
            "error_rate": round(counters.get("errors", 0) / max(1, counters.get("requests_received", 0)), 6),
            "latency": latency,
        }


class SZCFMPipeline:
    def __init__(
        self,
        config: SZCFMConfig | None = None,
        *,
        backend: InferenceBackend | None = None,
        masker: PIIMasker | None = None,
        publisher: EventProducer | None = None,
        forwarder: Forwarder | None = None,
        envelope_sink: EnvelopeSink | None = None,
    ) -> None:
        self.config = config or SZCFMConfig()
        cfg = self.config
        self.masker = masker or PIIMasker()
        self.extractor = FeatureExtractor(cfg.max_request_bytes)
        self.ring = RequestStateRing(cfg.ring_capacity, cfg.ring_slot_bytes, int(cfg.stale_after_s * 1e9))
        self.executor = AsyncInferenceExecutor(
            backend or DevelopmentHeuristicBackend(),
            workers=cfg.inference_workers,
            max_in_flight=cfg.max_in_flight,
        )
        self.policy = PolicyEngine(cfg)
        self.metrics = SZCFMMetrics()
        self._publisher = publisher
        self._forwarder = forwarder
        self._envelope_sink = envelope_sink

    # --- Stage A -----------------------------------------------------------------

    def parse(self, raw: bytes) -> SZCFMEvent:
        self.metrics.count("requests_received")
        if len(raw) > self.config.max_request_bytes:
            self.metrics.count("rejected_oversized")
            raise PayloadTooLargeError(
                f"request is {len(raw)} bytes; limit is {self.config.max_request_bytes}"
            )
        try:
            data = json.loads(raw, parse_constant=_reject_constant)
        except MalformedRequestError:
            self.metrics.count("rejected_malformed")
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.metrics.count("rejected_malformed")
            raise MalformedRequestError(f"body is not valid JSON: {exc.msg if hasattr(exc, 'msg') else exc}") from None
        except RecursionError:
            self.metrics.count("rejected_malformed")
            raise MalformedRequestError("body nesting is too deep") from None
        if not isinstance(data, dict):
            self.metrics.count("rejected_malformed")
            raise MalformedRequestError("body must be a JSON object")
        return self.validate(data)

    def validate(self, data: dict[str, Any]) -> SZCFMEvent:
        try:
            return SZCFMEvent.model_validate(data)
        except ValidationError as exc:
            self.metrics.count("rejected_invalid")
            raise EventValidationError(_safe_errors(exc)) from None

    # --- Stages B-H ------------------------------------------------------------------

    def process(
        self,
        event: SZCFMEvent,
        *,
        header_correlation_id: str | None = None,
        started_ns: int | None = None,
        validation_ns: int = 0,
    ) -> SZCFMResult:
        start = started_ns if started_ns is not None else time.perf_counter_ns()
        cfg = self.config
        body_corr = event.payload.correlation_id
        if header_correlation_id and body_corr and header_correlation_id != body_corr:
            self.metrics.count("rejected_correlation_conflict")
            raise CorrelationConflictError("X-Correlation-ID header does not match payload.correlation_id")
        event_id = str(uuid.uuid4())
        request_id = event.payload.request_id or f"req-{uuid.uuid4().hex}"
        correlation_id = body_corr or header_correlation_id or f"corr-{uuid.uuid4().hex}"
        ingested_at_ns = time.time_ns()

        # B. masking: authorization is reduced to presence + scheme here and never stored.
        t = time.perf_counter_ns()
        auth_header = event.metadata.authorization
        auth_scheme = auth_header.split(" ", 1)[0] if auth_header else ""
        masked_payload = self.masker.mask(event.payload.model_dump(mode="json", exclude_none=True))
        masked_headers = self.masker.mask(event.metadata.headers)
        masking_ns = time.perf_counter_ns() - t

        # D. features
        t = time.perf_counter_ns()
        features = self.extractor.extract(event)
        feature_ns = time.perf_counter_ns() - t

        # E. slot (RingFullError propagates; nothing to release yet)
        ticket = self.ring.allocate(request_id, correlation_id)
        try:
            result = self._process_owned(
                event, ticket, event_id, ingested_at_ns, masked_payload, masked_headers,
                auth_scheme, features, start, validation_ns, masking_ns, feature_ns,
            )
        except Exception:
            self.metrics.count("errors")
            log.exception("szcfm pipeline error event_id=%s request_id=%s", event_id, request_id)
            raise
        finally:
            released = self.ring.release(ticket)
        result.request_state.released = released
        return result

    def _process_owned(
        self, event, ticket, event_id, ingested_at_ns, masked_payload, masked_headers,
        auth_scheme, features, start, validation_ns, masking_ns, feature_ns,
    ) -> SZCFMResult:
        cfg = self.config
        request_id, correlation_id = ticket.request_id, ticket.correlation_id

        # C. protobuf serialization into the slot
        t = time.perf_counter_ns()
        serialization = self._serialize(
            event, ticket, event_id, ingested_at_ns, masked_payload, masked_headers, auth_scheme, features.values
        )
        serialization_ns = time.perf_counter_ns() - t

        # H1. deterministic rules
        outcome = self.policy.evaluate_deterministic(event, features.payload_scan.matched_signatures)
        deterministic = outcome.decision
        if not serialization.success:
            outcome.add("ENVELOPE_SERIALIZATION_FAILED", "policy", "CHALLENGE", serialization.error or "serialization failed")

        # F. asynchronous inference
        ai_result: InferenceResult | None = None
        ai_status: str
        ai_wait_ns: int | None = None
        if deterministic == "BLOCK":
            ai_status = InferenceStatus.SKIPPED.value
        else:
            self.ring.mark_ai_pending(ticket)
            task = InferenceTask(ticket.slot_index, ticket.generation, request_id, correlation_id, features.values)
            t = time.perf_counter_ns()
            if self.executor.submit(task, self._on_inference_result) is None:
                self.ring.mark_ai_unavailable(ticket, "REJECTED_BACKPRESSURE")
                self.metrics.count("ai_rejected_backpressure")
                ai_status = InferenceStatus.REJECTED_BACKPRESSURE.value
            else:
                state, ai_result = self.ring.wait_for_ai(ticket, cfg.ai_timeout_s)
                ai_wait_ns = time.perf_counter_ns() - t
                if state is SlotState.AI_COMPLETED and ai_result is not None:
                    ai_status = ai_result.status.value
                else:
                    self.metrics.count("ai_timeouts")
                    ai_status = InferenceStatus.TIMEOUT.value

        # G. context fusion
        t = time.perf_counter_ns()
        feature_map = features.as_dict()
        ctx_risk = context_risk(feature_map)
        fused_risk: float | None = None
        failure_mode: str | None = None
        usable = (
            ai_result is not None
            and ai_result.status is InferenceStatus.COMPLETED
            and not (ai_result.simulated and cfg.simulated_ai_policy == "unavailable")
        )
        if usable:
            fused_risk = fuse(ai_result.risk_score, ctx_risk, cfg.ai_weight)
            self.policy.apply_model(outcome, fused_risk, ai_result.simulated)
        elif ai_status != InferenceStatus.SKIPPED.value:
            status_for_policy = ai_status
            if ai_result is not None and ai_result.status is InferenceStatus.COMPLETED:
                status_for_policy = "SIMULATED_NOT_ACCEPTED"
            failure_mode = self.policy.apply_ai_unavailable(outcome, event.metadata.path, status_for_policy)
        contributing = ["deterministic_rules", "security_context" if event.payload.security_context else "security_context:missing"]
        if usable:
            contributing += ["ai_risk_score", "context_risk"]
        elif failure_mode:
            contributing.append(f"ai_failure_mode:{failure_mode}")
        fusion_ns = time.perf_counter_ns() - t

        # H2. decision, committed once
        t = time.perf_counter_ns()
        decision = outcome.decision
        committed = self.ring.commit_decision(ticket, decision)
        if not committed:
            self.metrics.count("double_decision_rejected")
        slot_state = self.ring.state_of(ticket)
        decision_ns = time.perf_counter_ns() - t

        forwarding = self._forward(event, decision, request_id, correlation_id)
        publication = self._publish(event, event_id, request_id, correlation_id, masked_payload, decision, ai_status)

        total_ns = time.perf_counter_ns() - start
        self._record(decision, ai_status, total_ns, serialization_ns, feature_ns, ai_result, ai_wait_ns, fusion_ns)
        log.info(
            "szcfm decision event_id=%s request_id=%s correlation_id=%s decision=%s reasons=%s ai_status=%s total_ms=%.3f",
            event_id, request_id, correlation_id, decision,
            ",".join(r.code for r in outcome.reasons), ai_status, total_ns / 1e6,
        )
        if ai_result is not None:
            ai_view = AIAssessmentResult(
                status=ai_status,
                model_version=ai_result.model_version,
                simulated=ai_result.simulated,
                risk_score=ai_result.risk_score,
                threat_level=threat_level(ai_result.risk_score, cfg) if ai_result.risk_score is not None else None,
                error=ai_result.error,
            )
        else:
            ai_view = AIAssessmentResult(
                status=ai_status,
                model_version=self.executor.backend.model_version,
                simulated=self.executor.backend.is_simulated,
            )
        return SZCFMResult(
            event_id=event_id,
            request_id=request_id,
            correlation_id=correlation_id,
            source=event.source,
            ingested_at=datetime.fromtimestamp(ingested_at_ns / 1e9, timezone.utc).isoformat(),
            event_status="COMPLETED" if committed else "FAILED",
            masked_payload=masked_payload,
            serialization=serialization,
            feature_extraction=FeatureExtractionResult(
                schema_version=FEATURE_SCHEMA_VERSION,
                feature_count=len(features.values),
                feature_names=list(features.names),
                vector=[round(v, 6) for v in features.values],
                missing_groups=list(features.missing_groups),
            ),
            request_state=RequestStateResult(
                slot_allocated=True,
                slot_index=ticket.slot_index,
                generation=ticket.generation,
                state=slot_state.value if slot_state else "RECLAIMED",
                released=False,
                ring_occupancy=self.ring.occupancy,
                ring_capacity=self.ring.capacity,
            ),
            ai_assessment=ai_view,
            context_fusion=ContextFusionResult(
                status="COMPLETED" if usable else ("SKIPPED" if ai_status == "SKIPPED" else "DEGRADED"),
                request_context_matched=usable or (ai_result is not None),
                stale_result=False,
                duplicate_result=False,
                delivery_outcome=DeliveryOutcome.ACCEPTED.value if ai_result is not None else None,
                context_risk=round(ctx_risk, 6),
                fused_risk=None if fused_risk is None else round(fused_risk, 6),
                formula=FUSION_FORMULA,
                contributing_inputs=contributing,
            ),
            security_decision=SecurityDecisionResult(
                decision=decision,
                reason_codes=[r.code for r in outcome.reasons],
                reasons=outcome.reasons,
                deterministic_decision=deterministic,
                ai_failure_mode=failure_mode,
                committed=committed,
            ),
            forwarding=forwarding,
            publication=publication,
            pipeline_metrics=PipelineMetrics(
                execution_time_ms=_ms(total_ns),
                validation_time_ms=_ms(validation_ns),
                masking_time_ms=_ms(masking_ns),
                serialization_time_ms=_ms(serialization_ns),
                feature_extraction_time_ms=_ms(feature_ns),
                ai_inference_time_ms=_ms(ai_result.latency_ns) if ai_result else None,
                ai_wait_time_ms=_ms(ai_wait_ns),
                context_fusion_time_ms=_ms(fusion_ns),
                decision_time_ms=_ms(decision_ns),
            ),
        )

    # --- helpers ---------------------------------------------------------------------

    def _on_inference_result(self, result: InferenceResult) -> None:
        outcome = self.ring.deliver_ai_result(
            result.slot_index, result.generation, result.request_id, result.correlation_id, result
        )
        self.metrics.count(f"ai_delivery_{outcome.value.lower()}")
        if outcome is not DeliveryOutcome.ACCEPTED:
            log.info(
                "szcfm discarded AI result request_id=%s correlation_id=%s outcome=%s",
                result.request_id, result.correlation_id, outcome.value,
            )

    def _serialize(self, event, ticket, event_id, ingested_at_ns, masked_payload, masked_headers, auth_scheme, vector) -> SerializationResult:
        try:
            envelope = SZCFMEnvelope(
                schema_version=ENVELOPE_SCHEMA_VERSION,
                event_id=event_id,
                request_id=ticket.request_id,
                correlation_id=ticket.correlation_id,
                source=event.source,
                ingested_at_unix_ns=ingested_at_ns,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                feature_vector=vector,
                has_prior_threat_score=event.threat_score is not None,
                prior_threat_score=event.threat_score or 0.0,
            )
            meta = event.metadata
            envelope.metadata.method = meta.method
            envelope.metadata.path = meta.path
            envelope.metadata.content_type = meta.content_type
            envelope.metadata.client_role = meta.client_role or ""
            envelope.metadata.authorization_present = bool(meta.authorization)
            envelope.metadata.authorization_scheme = auth_scheme
            envelope.metadata.headers.update({k: str(v) for k, v in masked_headers.items()})
            envelope.masked_payload.update(masked_payload)
            data = envelope.SerializeToString(deterministic=True)
            view = self.ring.store_envelope(ticket, data)
            if self._envelope_sink is not None:
                self._envelope_sink(view, ticket)
            return SerializationResult(
                success=True,
                bytes_size=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                schema_version=ENVELOPE_SCHEMA_VERSION,
                message_type=ENVELOPE_MESSAGE_TYPE,
            )
        except Exception as exc:
            self.metrics.count("serialization_failures")
            return SerializationResult(
                success=False,
                bytes_size=0,
                schema_version=ENVELOPE_SCHEMA_VERSION,
                message_type=ENVELOPE_MESSAGE_TYPE,
                error=f"{type(exc).__name__}: {exc}"[:300],
            )

    def _forward(self, event, decision, request_id, correlation_id) -> ForwardingResult:
        if decision != "ALLOW":
            return ForwardingResult(attempted=False, status="NOT_PERMITTED", detail=f"decision {decision}")
        if self._forwarder is None:
            return ForwardingResult(attempted=False, status="NOT_CONFIGURED")
        try:
            self._forwarder(event, request_id, correlation_id)
        except Exception as exc:
            self.metrics.count("forwarding_failures")
            return ForwardingResult(attempted=True, status="FAILED", detail=type(exc).__name__)
        return ForwardingResult(attempted=True, status="FORWARDED")

    def _publish(self, event, event_id, request_id, correlation_id, masked_payload, decision, ai_status) -> dict[str, Any]:
        if self._publisher is None:
            return {"status": "NOT_CONFIGURED"}
        raw = RawEvent(
            event_id=event_id,
            source=event.source,
            payload={
                **masked_payload,
                "_szcfm": {
                    "request_id": request_id,
                    "correlation_id": correlation_id,
                    "method": event.metadata.method,
                    "path": event.metadata.path,
                    "decision": decision,
                    "ai_status": ai_status,
                },
            },
        )
        try:
            self._publisher.publish(raw)
        except Exception as exc:
            self.metrics.count("publication_failures")
            return {"status": "FAILED", "error": type(exc).__name__}
        return {"status": "PUBLISHED"}

    def _record(self, decision, ai_status, total_ns, serialization_ns, feature_ns, ai_result, ai_wait_ns, fusion_ns) -> None:
        m = self.metrics
        m.count("requests_processed")
        m.count(f"decision_{decision.lower()}")
        m.count(f"ai_status_{ai_status.lower()}")
        m.sample("total", total_ns)
        m.sample("serialization", serialization_ns)
        m.sample("feature_extraction", feature_ns)
        m.sample("ai_inference", ai_result.latency_ns if ai_result else None)
        m.sample("ai_wait", ai_wait_ns)
        m.sample("context_fusion", fusion_ns)

    def stats(self) -> dict[str, Any]:
        snapshot = self.metrics.snapshot()
        snapshot["ring"] = self.ring.stats()
        snapshot["inference"] = {
            "in_flight": self.executor.in_flight,
            "max_in_flight": self.executor.max_in_flight,
            "rejected_backpressure": self.executor.rejected,
            "model_version": self.executor.backend.model_version,
            "simulated": self.executor.backend.is_simulated,
        }
        return snapshot

    def close(self) -> None:
        self.executor.shutdown()
