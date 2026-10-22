import redis
from fastapi.testclient import TestClient

from api import routes
from api.routes import app
from src.policy_verifier.models import RequestDecisionReport
from src.policy_verifier.request_cache import KEY_PREFIX, RequestDecisionCache

client = TestClient(app)

POLICY = {
    "policy_id": "cache-p1", "subject": "Administrators", "action": "Delete",
    "resource": "UserAccounts", "effect": "Allow",
    "raw_text": "Administrators can delete user accounts.",
}


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.ttls = {}
        self.calls = 0

    def get(self, key):
        self.calls += 1
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.calls += 1
        self.store[key] = value
        self.ttls[key] = ttl


class DownRedis:
    def __init__(self):
        self.calls = 0

    def get(self, key):
        self.calls += 1
        raise redis.ConnectionError("connection refused")

    def setex(self, key, ttl, value):
        self.calls += 1
        raise redis.TimeoutError("timed out")


def _post(roles, policies=(POLICY,)):
    return client.post("/api/v1/verify-request", json={
        "policies": list(policies),
        "request": {"subject_id": "s1", "roles": roles, "action": "Delete", "resource": "UserAccounts"},
    })


def _counting_engine(monkeypatch):
    calls = []
    original = routes.pipeline.engine.check_request

    def check_request(*args, **kwargs):
        calls.append(kwargs.get("roles"))
        return original(*args, **kwargs)

    monkeypatch.setattr(routes.pipeline.engine, "check_request", check_request)
    return calls


def test_miss_runs_z3_and_stores_json_with_one_hour_ttl(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(routes, "request_cache", RequestDecisionCache(fake))
    z3_calls = _counting_engine(monkeypatch)

    first = _post(["Administrators"])
    assert first.status_code == 200
    assert first.json()["decision"] == "ALLOWED"
    assert len(z3_calls) == 1
    [(key, value)] = fake.store.items()
    assert key.startswith(KEY_PREFIX) and len(key) == len(KEY_PREFIX) + 32
    assert fake.ttls[key] == 3600
    assert RequestDecisionReport.model_validate_json(value).decision.value == "ALLOWED"

    second = _post(["Administrators"])
    assert len(z3_calls) == 1  # served from the cache, Z3 not called again
    assert second.json() == first.json()


def test_roles_and_policies_are_part_of_the_key(monkeypatch):
    monkeypatch.setattr(routes, "request_cache", RequestDecisionCache(FakeRedis()))
    z3_calls = _counting_engine(monkeypatch)

    assert _post(["Administrators"]).json()["decision"] == "ALLOWED"
    assert _post(["Guests"]).json()["decision"] == "NOT_DETERMINED"
    deny = dict(POLICY, policy_id="cache-p2", effect="Deny", interpretation="role_prohibition",
                raw_text="Administrators cannot delete user accounts.")
    assert _post(["Administrators"], policies=(deny,)).json()["decision"] == "DENIED"
    assert len(z3_calls) == 3


def test_cached_hit_does_not_repeat_unmatched_role_notes(monkeypatch):
    monkeypatch.setattr(routes, "request_cache", RequestDecisionCache(FakeRedis()))
    first = _post(["Administrator"]).json()["limitations"]
    second = _post(["Administrator"]).json()["limitations"]
    assert first == second
    assert sum("matched no policy role" in note for note in second) == 1


def test_redis_down_falls_through_to_z3_and_backs_off(monkeypatch):
    down = DownRedis()
    monkeypatch.setattr(routes, "request_cache", RequestDecisionCache(down, retry_after_seconds=60))
    z3_calls = _counting_engine(monkeypatch)

    for _ in range(3):
        response = _post(["Administrators"])
        assert response.status_code == 200
        assert response.json()["decision"] == "ALLOWED"
    assert len(z3_calls) == 3
    assert down.calls == 1  # after the first failure Redis is skipped, not retried per request


def test_unreadable_cache_entry_is_ignored(monkeypatch):
    fake = FakeRedis()
    cache = RequestDecisionCache(fake)
    monkeypatch.setattr(routes, "request_cache", cache)
    _post(["Administrators"])
    [key] = fake.store
    fake.store[key] = b"not json"
    z3_calls = _counting_engine(monkeypatch)
    assert _post(["Administrators"]).json()["decision"] == "ALLOWED"
    assert len(z3_calls) == 1


def test_solver_unknown_is_never_cached(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(routes, "request_cache", RequestDecisionCache(fake))

    class UnknownSolver:
        def add(self, *constraints):
            pass

        def check(self):
            import z3
            return z3.unknown

        def reason_unknown(self):
            return "forced"

    monkeypatch.setattr(routes.pipeline.engine, "_solver_factory", UnknownSolver)
    assert _post(["Administrators"]).json()["reason_code"] == "solver_unknown"
    assert fake.store == {}


def test_cache_is_disabled_for_the_test_run_by_default():
    assert routes.request_cache.enabled is False
