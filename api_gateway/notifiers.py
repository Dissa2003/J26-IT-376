from __future__ import annotations

import logging

from shared_contracts.interfaces import Notifier
from shared_contracts.models import Alert

log = logging.getLogger("notifier")


class LogNotifier(Notifier):
    """Stub connector. Add Slack/email/webhook notifiers implementing Notifier."""

    def notify(self, alert: Alert) -> None:
        log.warning("ALERT %s [%s] %s", alert.alert_id, alert.severity.value, alert.message)