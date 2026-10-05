"""Abstract interfaces. Each member implements the ones for their module; others depend only on these."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from .models import Alert, FeatureVector, Prediction, RawEvent


# --- Member 1: ingestion ---
class EventProducer(ABC):
    @abstractmethod
    def publish(self, event: RawEvent) -> None: ...


class EventConsumer(ABC):
    @abstractmethod
    def subscribe(self, handler: Callable[[RawEvent], None]) -> None: ...


# --- Member 2: processing ---
class SchemaValidator(ABC):
    @abstractmethod
    def validate(self, event: RawEvent) -> None:
        """Raise ValueError if the event is malformed."""


class FeatureTransformer(ABC):
    @abstractmethod
    def transform(self, event: RawEvent) -> FeatureVector: ...


# --- Member 3: engine ---
class InferenceModel(ABC):
    @abstractmethod
    def score(self, features: FeatureVector) -> float:
        """Return a score in [0, 1]."""


class PredictionService(ABC):
    @abstractmethod
    def predict(self, features: FeatureVector) -> Prediction: ...


# --- Member 4: gateway ---
class Notifier(ABC):
    @abstractmethod
    def notify(self, alert: Alert) -> None: ...


# --- Cross-module clients (implemented by the caller side, e.g. HTTP) ---
class ProcessingClient(ABC):
    @abstractmethod
    def process(self, event: RawEvent) -> Prediction: ...


class InferenceClient(ABC):
    @abstractmethod
    def predict(self, features: FeatureVector) -> Prediction: ...


class AlertClient(ABC):
    @abstractmethod
    def send_alert(self, alert: Alert) -> None: ...