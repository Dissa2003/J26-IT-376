# Output contracts — Formal Verification

This folder documents what the Formal Verification component (`api/` + `src/policy_verifier/`)
returns to other components. It describes **implemented behaviour** as of branch
`feature/policy_verification`; it has not yet been reviewed by the consuming members.

| File | Describes |
|---|---|
| `verify_request_response_schema.json` | JSON Schema (draft-07) for the HTTP 200 body of `POST /api/v1/verify-request` |

## Consumers

| Member | Component | Use of this output |
|---|---|---|
| Abeykoon K.G.T.D. | LogiShield (`tokenization_ai/`) | Verified rule information as an input signal (see `tokenization_ai/CONTRACT_PROPOSAL.md` §2–3) |
| Kumara | *To be confirmed* — not recorded in the repository | Expected to act on `decision` (for example, an Allow/Block verdict) |

## Calling the endpoint

```
POST /api/v1/verify-request
Content-Type: application/json
```

| Where | URL |
|---|---|
| Policy service directly | `http://127.0.0.1:8002/api/v1/verify-request` |
| Through the LogiShield dashboard proxy | `http://127.0.0.1:8010/api/v1/verify-request` |

Start both services with `scripts\start_demo.ps1`, or the policy service alone with `python -m api`.

### Request body

```json
{
  "policies": [
    {
      "policy_id": "p1",
      "subject": "Administrators",
      "action": "Delete",
      "resource": "UserAccounts",
      "effect": "Allow",
      "interpretation": "role_permission",
      "raw_text": "Administrators can delete user accounts."
    }
  ],
  "request": {
    "subject_id": "alice-01",
    "roles": ["Administrators"],
    "action": "Delete",
    "resource": "UserAccounts"
  }
}
```

| Field | Rules |
|---|---|
| `policies` | 1–100 policies. The easiest way to get correct policy objects is to take them from `POST /api/v1/verify-text` (`attributes` plus `raw_text`). |
| `policies[].raw_text` | If non-empty, it is re-parsed and must agree with `subject`, `action`, `resource`, `effect` and `interpretation`; otherwise HTTP 422. Supported sentence forms are listed in `src/policy_verifier/GRAMMAR.md`. |
| `policies[].interpretation` | `role_permission` (*X can …*), `only_role` (*only X can …*) or `role_prohibition` (*X cannot …*). Defaults to `role_permission`. |
| `request.roles` | Up to 32 roles. Treated as the subject's **complete** role assignment: any role not listed is assumed *not* held. |
| `request.action`, `request.resource` | Write names as single words in the same form as the policy fields (`Delete`, `UserAccounts`). Do **not** use spaces — see *Known issue* below. |

### Example response (real output for the body above)

```json
{
  "decision": "ALLOWED",
  "reason_code": "request_entailed",
  "explanation": "The request is entailed by the submitted policies and role assignment.",
  "consistency_query": { "status": "sat", "reason_unknown": null },
  "allow_query": { "status": "unsat", "reason_unknown": null },
  "deny_query": { "status": "sat", "reason_unknown": null },
  "constraints": [
    { "constraint_id": "p1:rule", "expression": "∀x (Administrators(x) → CanDelete(x, Useraccounts))", "kind": "rule", "source_policy_ids": ["p1"] },
    { "constraint_id": "request:role:Administrators", "expression": "Administrators(Alice-01)", "kind": "fact", "source_policy_ids": [] },
    { "constraint_id": "request:query", "expression": "CanDelete(Alice-01, Useraccounts)", "kind": "query", "source_policy_ids": [] },
    { "constraint_id": "request:negated-query", "expression": "¬CanDelete(Alice-01, Useraccounts)", "kind": "negated_query", "source_policy_ids": [] }
  ],
  "assumptions": [
    "Role-witness axiom enabled.",
    "Role-exclusivity axiom disabled for independent request checking."
  ],
  "execution_time_ms": 31.47,
  "limitations": [
    "The request roles are a complete role assignment for the subject.",
    "Policy conditions are outside the supported policy grammar."
  ]
}
```

`execution_time_ms` varies per call. Names in `constraints` are normalised (`Useraccounts`, `Alice-01`), so do not
compare them with your own spelling.

### Error responses (HTTP 422)

| `detail.code` | Meaning |
|---|---|
| `policy_raw_text_unsupported` | A policy's `raw_text` is outside the supported grammar. `detail.policy_id` names it. |
| `policy_field_mismatch` | A policy's fields disagree with its `raw_text`. `detail.policies[]` lists the policy ids and differing fields. |
| *(no code; `detail` is a list)* | Standard FastAPI validation error, e.g. an empty `policies` list or a missing `subject_id`. |

## What each decision means downstream

Branch on `decision` and `reason_code`. `explanation` is for people and may change wording.

| `decision` | `reason_code` | Formal meaning | Suggested downstream handling |
|---|---|---|---|
| `ALLOWED` | `request_entailed` | The policies and role assignment **logically entail** the request (`allow_query` is unsat). | Safe to treat as *permitted by policy*. |
| `DENIED` | `request_prohibited` | The policies and role assignment entail that the request is **not** permitted (`deny_query` is unsat). | Treat as *forbidden by policy* (e.g. Block). |
| `NOT_DETERMINED` | `not_covered_by_policies` | Neither the request nor its negation follows. No submitted policy decides this subject/action/resource. | Policy gap, **not** permission. The consumer chooses its own default; fail-closed (treat as not allowed) is the safe choice. |
| `NOT_DETERMINED` | `policy_set_contradictory` | The policies are inconsistent for this subject's roles, e.g. one role grants and another prohibits the same action. | A policy-authoring error. Do not allow; surface it for correction. |
| `NOT_DETERMINED` | `solver_unknown` | Z3 could not decide one of the queries; `*_query.reason_unknown` carries Z3's reason. | Indeterminate. Fail closed; retrying with the same input is unlikely to change the result. |

The pairing is fixed: `ALLOWED` always comes with `request_entailed`, `DENIED` with `request_prohibited`, and
`NOT_DETERMINED` with one of the other three. The schema enforces this.

`only_role` policies (*only X can …*) state a **necessary** condition, not a grant. Holding the role alone gives
`NOT_DETERMINED`; lacking it gives `DENIED`.

## Limitations consumers should know

- **No event correlation.** The response carries no `event_id`; the caller must associate it with the API event
  it checked (`tokenization_ai/CONTRACT_PROPOSAL.md` Q2.2).
- **No mapping from HTTP requests.** The caller must supply `subject_id`, `roles`, `action` and `resource`;
  this component does not derive them from method, path or headers (Q2.3).
- **No attribute conditions.** Policies with conditions (`if`, `when`, `with` …) are rejected by the grammar.
- **Process-local.** Nothing from `verify-request` is stored; the rule cache and records endpoints are in-memory.

## Known issue: spaced names in the request

The service normalises names twice, and spaced and single-word spellings come out differently. With a
policy on `UserAccounts`:

| `request.resource` | Result |
|---|---|
| `UserAccounts`, `Useraccounts`, `useraccounts` | `ALLOWED` |
| `user accounts` | `NOT_DETERMINED` / `not_covered_by_policies` — **incorrect** |

Until this is fixed in `api/routes.py`, send `action` and `resource` as single words without spaces.
