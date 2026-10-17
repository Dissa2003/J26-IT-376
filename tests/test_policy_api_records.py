from fastapi.testclient import TestClient

from api.routes import app


def test_verification_records_and_cache_are_explicitly_process_local():
    client = TestClient(app)
    response = client.post("/api/v1/verify-text", json={
        "policy_text": "Administrators can read user profiles.",
        "policy_id": "records-test",
    })
    assert response.status_code == 200
    records = client.get("/api/v1/verification-records")
    assert records.status_code == 200
    assert records.json()[0]["policy_id"] == "records-test"
    cached = client.get("/api/v1/rule-cache")
    assert cached.status_code == 200
    assert any(item["policy_id"] == "records-test" for item in cached.json())


def test_conflict_endpoint_returns_recorded_conflicts():
    client = TestClient(app)
    client.post("/api/v1/verify-policies", json=[
        {
            "policy_id": "conflict-allow", "subject": "Administrators",
            "action": "Read", "resource": "UserProfiles", "effect": "Allow",
            "raw_text": "allow",
        },
        {
            "policy_id": "conflict-deny", "subject": "Administrators",
            "action": "Read", "resource": "UserProfiles", "effect": "Deny",
            "raw_text": "deny",
        },
    ])
    conflicts = client.get("/api/v1/conflicts")
    assert conflicts.status_code == 200
    assert any(item["conflict_id"] == "conflict-allow:conflict-deny" for item in conflicts.json())
