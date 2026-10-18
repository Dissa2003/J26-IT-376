# LogiShield Evaluation Ground-Truth Review

## Scope and provenance

This is a separate, AI-authored mathematical review of the 37 records in
`src/policy_verifier/evaluation_dataset.json`. It was written from the policy
text and the documented grammar in `src/policy_verifier/GRAMMAR.md`.

This review did **not** call the production `PolicyParser`, `to_fol`,
`infer`, `VerificationPipeline`, `PolicyVerificationEngine`, or
`ConflictDetector` to generate reference answers. It did not modify the
dataset or its expected labels. It is not independent human validation.
Every item marked `appears correct` still requires supervisor or second-reviewer
confirmation before being used as research ground truth.

## Reference conventions

For a policy with role `R`, action `A`, and resource `S`:

- `R can A S` means `∀x (R(x) → CanA(x, S))`.
- `Only R can A S` means `∀x (CanA(x, S) → R(x))`.
- `R cannot A S` means `∀x (R(x) → ¬CanA(x, S))`.

For an entailment query, the expected result is:

- `UNSAT` when `K ∧ ¬C` is unsatisfiable, so `K ⊨ C`.
- `SAT` when a countermodel to the conclusion exists.
- `UNKNOWN` only when the solver cannot determine the query.

For a consistency query, `UNSAT` means the policy constraints cannot all hold
together. This is different from entailment `UNSAT`.

The current ordinary records do not supply an independent request. Their
verification request is generated from the policy antecedent, so the premises
are effectively the instantiated antecedent and instantiated implication:
`R(Alice)` and `R(Alice) → C(Alice)`, with conclusion `C(Alice)`. Modus Ponens
therefore succeeds by construction. This is a limitation of the case design,
not independent evidence of authorization-decision accuracy.

## Summary

| Review category | Cases |
|---|---:|
| Appear mathematically correct under the documented grammar | 32 |
| Unsupported inputs correctly treated as unsupported | 5 |
| Cases with a likely label/query-design limitation | 31 ordinary entailment cases |
| Cases requiring supervisor or second-reviewer confirmation | All 37 |
| Cases that appear to require dataset-label revision from this review alone | 0 |

The 32 supported records are internally consistent with the stated grammar,
but most are tautological demonstrations because the request fact is generated
from the policy antecedent. `consistency-01`, `nonentail-01`, and
`nonentail-02` are the most informative solver cases, although their reference
premises and query construction should still receive independent review.

## Case-by-case independent review

### Access-control allow cases

#### `allow-01`

- **Input:** `Administrators can read user profiles.`
- **Structured attributes:** subject `Administrators`; action `Read`; resource
  `UserProfiles`; effect `Allow`; interpretation `role_permission`; conditions
  empty.
- **FOL:** `∀x (Administrators(x) → CanRead(x, UserProfiles))`.
- **Verification query:** entailment. Under the current case convention,
  `K = {Administrators(Alice), Administrators(Alice) → CanRead(Alice, UserProfiles)}`
  and `C = CanRead(Alice, UserProfiles)`.
- **Expected result:** `UNSAT` for `K ∧ ¬C`; the implication and antecedent
  force the conclusion.
- **Conflict:** no. There is only one allow rule and no opposite rule with the
  same subject, action, resource, and conditions.
- **Grammar:** supported.
- **Assessment:** appears correct, but the entailment succeeds by construction.
- **Review status:** second review required.

#### `allow-02`

- **Input:** `Auditors can read audit logs.`
- **Structured attributes:** `Auditors`, `Read`, `AuditLogs`, `Allow`,
  `role_permission`, no conditions.
- **FOL:** `∀x (Auditors(x) → CanRead(x, AuditLogs))`.
- **Query:** entailment with `K = {Auditors(Alice), Auditors(Alice) → CanRead(Alice, AuditLogs)}`
  and `C = CanRead(Alice, AuditLogs)`.
- **Expected result:** `UNSAT`; Modus Ponens makes the negated conclusion
  impossible.
- **Conflict:** no.
- **Grammar:** supported.
- **Assessment:** appears correct; independent request coverage is absent.
- **Review status:** second review required.

#### `allow-03`

- **Input:** `Operators can restart services.`
- **Structured attributes:** `Operators`, `Restart`, `Services`, `Allow`,
  `role_permission`, no conditions.
- **FOL:** `∀x (Operators(x) → CanRestart(x, Services))`.
- **Query:** entailment; `K = {Operators(Alice), Operators(Alice) → CanRestart(Alice, Services)}`;
  `C = CanRestart(Alice, Services)`.
- **Expected result:** `UNSAT`; Modus Ponens.
- **Conflict:** no.
- **Grammar:** supported.
- **Assessment:** appears correct; construction is tautological.
- **Review status:** second review required.

#### `allow-04`

- **Input:** `Support agents can view tickets.`
- **Structured attributes:** `SupportAgents`, `View`, `Tickets`, `Allow`,
  `role_permission`, no conditions.
- **FOL:** `∀x (SupportAgents(x) → CanView(x, Tickets))`.
- **Query:** entailment; `K = {SupportAgents(Alice), SupportAgents(Alice) → CanView(Alice, Tickets)}`;
  `C = CanView(Alice, Tickets)`.
- **Expected result:** `UNSAT`; Modus Ponens.
- **Conflict:** no.
- **Grammar:** supported.
- **Assessment:** appears correct; multiword role normalization is not an
  independent semantic test.
- **Review status:** second review required.

#### `allow-05`

- **Input:** `Data stewards can export reports.`
- **Structured attributes:** `DataStewards`, `Export`, `Reports`, `Allow`,
  `role_permission`, no conditions.
- **FOL:** `∀x (DataStewards(x) → CanExport(x, Reports))`.
- **Query:** entailment; `K = {DataStewards(Alice), DataStewards(Alice) → CanExport(Alice, Reports)}`;
  `C = CanExport(Alice, Reports)`.
- **Expected result:** `UNSAT`; Modus Ponens.
- **Conflict:** no.
- **Grammar:** supported.
- **Assessment:** appears correct; still needs independent review of canonicalization.
- **Review status:** second review required.

#### `allow-06`

- **Input:** `Administrators can delete user accounts.`
- **Structured attributes:** `Administrators`, `Delete`, `UserAccounts`, `Allow`,
  `role_permission`, no conditions.
- **FOL:** `∀x (Administrators(x) → CanDelete(x, UserAccounts))`.
- **Query:** entailment; `K = {Administrators(Alice), Administrators(Alice) → CanDelete(Alice, UserAccounts)}`;
  `C = CanDelete(Alice, UserAccounts)`.
- **Expected result:** `UNSAT`; Modus Ponens.
- **Conflict:** no in the single-policy case.
- **Grammar:** supported.
- **Assessment:** formula and result are correct under the stated semantics.
  This is the clearest example of the request fact being generated from the
  antecedent.
- **Review status:** second review required.

#### `allow-07`

- **Input:** `Reviewers can approve documents.`
- **Structured attributes:** `Reviewers`, `Approve`, `Documents`, `Allow`,
  `role_permission`, no conditions.
- **FOL:** `∀x (Reviewers(x) → CanApprove(x, Documents))`.
- **Query:** entailment; `K = {Reviewers(Alice), Reviewers(Alice) → CanApprove(Alice, Documents)}`;
  `C = CanApprove(Alice, Documents)`.
- **Expected result:** `UNSAT`; Modus Ponens.
- **Conflict:** no.
- **Grammar:** supported.
- **Assessment:** appears correct; no unseen wording or request variation.
- **Review status:** second review required.

#### `allow-08`

- **Input:** `Developers can deploy applications.`
- **Structured attributes:** `Developers`, `Deploy`, `Applications`, `Allow`,
  `role_permission`, no conditions.
- **FOL:** `∀x (Developers(x) → CanDeploy(x, Applications))`.
- **Query:** entailment; `K = {Developers(Alice), Developers(Alice) → CanDeploy(Alice, Applications)}`;
  `C = CanDeploy(Alice, Applications)`.
- **Expected result:** `UNSAT`; Modus Ponens.
- **Conflict:** no.
- **Grammar:** supported.
- **Assessment:** appears correct; construction remains tautological.
- **Review status:** second review required.

### Only-role cases

#### `only-01`

- **Input:** `Only administrators can delete user accounts.`
- **Structured attributes:** `Administrators`, `Delete`, `UserAccounts`, `Allow`,
  `only_role`, no conditions.
- **FOL:** `∀x (CanDelete(x, UserAccounts) → Administrators(x))`.
- **Query:** entailment under the current generated-fact convention:
  `K = {CanDelete(Alice, UserAccounts), CanDelete(Alice, UserAccounts) → Administrators(Alice)}`;
  `C = Administrators(Alice)`.
- **Expected result:** `UNSAT`; the capability fact and implication force the
  role conclusion.
- **Conflict:** no.
- **Grammar:** supported.
- **Assessment:** implication direction is correct. The current generated
  antecedent is a capability, so the case does not test an unauthorized role
  attempting the action.
- **Review status:** supervisor review required.

#### `only-02`

- **Input:** `Only auditors can read audit logs.`
- **Structured attributes:** `Auditors`, `Read`, `AuditLogs`, `Allow`,
  `only_role`, no conditions.
- **FOL:** `∀x (CanRead(x, AuditLogs) → Auditors(x))`.
- **Query:** entailment; `K = {CanRead(Alice, AuditLogs), CanRead(Alice, AuditLogs) → Auditors(Alice)}`;
  `C = Auditors(Alice)`.
- **Expected result:** `UNSAT`.
- **Conflict:** no.
- **Grammar:** supported.
- **Assessment:** formula is correct; query construction does not test a
  non-auditor counterexample.
- **Review status:** supervisor review required.

#### `only-03`

- **Input:** `Only operators can restart services.`
- **Structured attributes:** `Operators`, `Restart`, `Services`, `Allow`,
  `only_role`, no conditions.
- **FOL:** `∀x (CanRestart(x, Services) → Operators(x))`.
- **Query:** entailment; `K = {CanRestart(Alice, Services), CanRestart(Alice, Services) → Operators(Alice)}`;
  `C = Operators(Alice)`.
- **Expected result:** `UNSAT`.
- **Conflict:** no.
- **Grammar:** supported.
- **Assessment:** formula appears correct; the case needs a separately authored
  capability fact for a meaningful authorization test.
- **Review status:** supervisor review required.

#### `only-04`

- **Input:** `Only support agents can view tickets.`
- **Structured attributes:** `SupportAgents`, `View`, `Tickets`, `Allow`,
  `only_role`, no conditions.
- **FOL:** `∀x (CanView(x, Tickets) → SupportAgents(x))`.
- **Query:** entailment; `K = {CanView(Alice, Tickets), CanView(Alice, Tickets) → SupportAgents(Alice)}`;
  `C = SupportAgents(Alice)`.
- **Expected result:** `UNSAT`.
- **Conflict:** no.
- **Grammar:** supported.
- **Assessment:** implication direction is correct; query is constructed to
  satisfy it.
- **Review status:** supervisor review required.

### Prohibition cases

#### `deny-01`

- **Input:** `Administrators cannot delete user accounts.`
- **Structured attributes:** `Administrators`, `Delete`, `UserAccounts`, `Deny`,
  `role_prohibition`, no conditions.
- **FOL:** `∀x (Administrators(x) → ¬CanDelete(x, UserAccounts))`.
- **Query:** entailment; `K = {Administrators(Alice), Administrators(Alice) → ¬CanDelete(Alice, UserAccounts)}`;
  `C = ¬CanDelete(Alice, UserAccounts)`.
- **Expected result:** `UNSAT`; the negative conclusion follows from the
  supplied role fact and prohibition rule.
- **Conflict:** no single-policy conflict.
- **Grammar:** supported.
- **Assessment:** appears correct; the request fact is again generated from the
  antecedent.
- **Review status:** second review required.

#### `deny-02`

- **Input:** `Auditors cannot modify audit logs.`
- **Structured attributes:** `Auditors`, `Modify`, `AuditLogs`, `Deny`,
  `role_prohibition`, no conditions.
- **FOL:** `∀x (Auditors(x) → ¬CanModify(x, AuditLogs))`.
- **Query:** entailment; `K = {Auditors(Alice), Auditors(Alice) → ¬CanModify(Alice, AuditLogs)}`;
  `C = ¬CanModify(Alice, AuditLogs)`.
- **Expected result:** `UNSAT`.
- **Conflict:** no.
- **Grammar:** supported.
- **Assessment:** appears correct under the documented negative-rule semantics.
- **Review status:** second review required.

#### `deny-03`

- **Input:** `Operators cannot stop services.`
- **Structured attributes:** `Operators`, `Stop`, `Services`, `Deny`,
  `role_prohibition`, no conditions.
- **FOL:** `∀x (Operators(x) → ¬CanStop(x, Services))`.
- **Query:** entailment; `K = {Operators(Alice), Operators(Alice) → ¬CanStop(Alice, Services)}`;
  `C = ¬CanStop(Alice, Services)`.
- **Expected result:** `UNSAT`.
- **Conflict:** no.
- **Grammar:** supported.
- **Assessment:** appears correct; no positive permission is supplied.
- **Review status:** second review required.

#### `deny-04`

- **Input:** `Guests cannot export reports.`
- **Structured attributes:** `Guests`, `Export`, `Reports`, `Deny`,
  `role_prohibition`, no conditions.
- **FOL:** `∀x (Guests(x) → ¬CanExport(x, Reports))`.
- **Query:** entailment; `K = {Guests(Alice), Guests(Alice) → ¬CanExport(Alice, Reports)}`;
  `C = ¬CanExport(Alice, Reports)`.
- **Expected result:** `UNSAT`.
- **Conflict:** no.
- **Grammar:** supported.
- **Assessment:** appears correct under the limited grammar.
- **Review status:** second review required.

#### `deny-05`

- **Input:** `Support agents cannot delete tickets.`
- **Structured attributes:** `SupportAgents`, `Delete`, `Tickets`, `Deny`,
  `role_prohibition`, no conditions.
- **FOL:** `∀x (SupportAgents(x) → ¬CanDelete(x, Tickets))`.
- **Query:** entailment; `K = {SupportAgents(Alice), SupportAgents(Alice) → ¬CanDelete(Alice, Tickets)}`;
  `C = ¬CanDelete(Alice, Tickets)`.
- **Expected result:** `UNSAT`.
- **Conflict:** no.
- **Grammar:** supported.
- **Assessment:** appears correct; request satisfaction is by construction.
- **Review status:** second review required.

### Direct conflict and scope cases

#### `conflict-01`

- **Input:** `Administrators can delete user accounts.` and
  `Administrators cannot delete user accounts.`
- **Structured attributes:** first rule is `Administrators`, `Delete`,
  `UserAccounts`, `Allow`, `role_permission`; second is the same scope with
  `Deny`, `role_prohibition`; conditions empty.
- **FOL formulas:** `∀x (Administrators(x) → CanDelete(x, UserAccounts))` and
  `∀x (Administrators(x) → ¬CanDelete(x, UserAccounts))`.
- **Query:** the existing label is entailment for the first policy, with
  `K = {Administrators(Alice), Administrators(Alice) → CanDelete(Alice, UserAccounts)}`
  and `C = CanDelete(Alice, UserAccounts)`. The two-policy set also supports a
  separate consistency query.
- **Expected result:** entailment `UNSAT`; separately, the full policy set is
  consistency `UNSAT` because it requires both a proposition and its negation.
- **Conflict:** yes; exact subject/action/resource/condition scope and opposite
  effects.
- **Grammar:** both inputs supported.
- **Assessment:** conflict and both mathematical results are correct, but one
  record combines an entailment label with a multi-policy conflict set. The
  query type should be independently reviewed.
- **Review status:** supervisor review required.

#### `conflict-02`

- **Input:** `Auditors can read audit logs.` and
  `Auditors cannot read audit logs.`
- **Structured attributes:** same `Auditors`, `Read`, `AuditLogs` scope with
  opposite `Allow` and `Deny` effects.
- **FOL formulas:** `∀x (Auditors(x) → CanRead(x, AuditLogs))` and
  `∀x (Auditors(x) → ¬CanRead(x, AuditLogs))`.
- **Query:** current first-policy entailment: `K = {Auditors(Alice), Auditors(Alice) → CanRead(Alice, AuditLogs)}`,
  `C = CanRead(Alice, AuditLogs)`.
- **Expected result:** entailment `UNSAT`; full-set consistency is separately
  `UNSAT`.
- **Conflict:** yes; exact scope and opposite effects.
- **Grammar:** both supported.
- **Assessment:** mathematically correct, but the record does not make the
  consistency-versus-entailment distinction explicit in its expected fields.
- **Review status:** supervisor review required.

#### `conflict-03`

- **Input:** `Operators can restart services.` and
  `Operators cannot restart services.`
- **Structured attributes:** same `Operators`, `Restart`, `Services` scope with
  opposite effects.
- **FOL formulas:** `∀x (Operators(x) → CanRestart(x, Services))` and
  `∀x (Operators(x) → ¬CanRestart(x, Services))`.
- **Query:** first-policy entailment with `K = {Operators(Alice), Operators(Alice) → CanRestart(Alice, Services)}`;
  `C = CanRestart(Alice, Services)`.
- **Expected result:** entailment `UNSAT`; full-set consistency is `UNSAT`.
- **Conflict:** yes.
- **Grammar:** both supported.
- **Assessment:** appears correct as a conflict example; query semantics need
  separate documentation.
- **Review status:** supervisor review required.

#### `scope-01`

- **Input:** `Administrators can delete user accounts.` and
  `Administrators cannot delete invoices.`
- **Structured attributes:** `Administrators/Delete/UserAccounts/Allow` and
  `Administrators/Delete/Invoices/Deny`.
- **FOL formulas:** `∀x (Administrators(x) → CanDelete(x, UserAccounts))` and
  `∀x (Administrators(x) → ¬CanDelete(x, Invoices))`.
- **Query:** first-policy entailment; `K = {Administrators(Alice), Administrators(Alice) → CanDelete(Alice, UserAccounts)}`;
  `C = CanDelete(Alice, UserAccounts)`.
- **Expected result:** `UNSAT`.
- **Conflict:** no; resources differ.
- **Grammar:** both supported.
- **Assessment:** appears correct under exact-scope conflict semantics.
- **Review status:** second review required.

#### `scope-02`

- **Input:** `Administrators can delete user accounts.` and
  `Auditors cannot delete user accounts.`
- **Structured attributes:** same action/resource but different roles and
  opposite effects.
- **FOL formulas:** `∀x (Administrators(x) → CanDelete(x, UserAccounts))` and
  `∀x (Auditors(x) → ¬CanDelete(x, UserAccounts))`.
- **Query:** first-policy entailment; `K = {Administrators(Alice), Administrators(Alice) → CanDelete(Alice, UserAccounts)}`;
  `C = CanDelete(Alice, UserAccounts)`.
- **Expected result:** `UNSAT`.
- **Conflict:** no under role-scoped semantics.
- **Grammar:** both supported.
- **Assessment:** appears correct; it assumes role predicates are disjoint
  only for conflict classification, not for logical satisfiability.
- **Review status:** supervisor review required for the role-overlap assumption.

#### `scope-03`

- **Input:** `Administrators can delete user accounts.` and
  `Administrators cannot read user accounts.`
- **Structured attributes:** same role/resource but different actions.
- **FOL formulas:** `∀x (Administrators(x) → CanDelete(x, UserAccounts))` and
  `∀x (Administrators(x) → ¬CanRead(x, UserAccounts))`.
- **Query:** first-policy entailment; `K = {Administrators(Alice), Administrators(Alice) → CanDelete(Alice, UserAccounts)}`;
  `C = CanDelete(Alice, UserAccounts)`.
- **Expected result:** `UNSAT`.
- **Conflict:** no; actions differ.
- **Grammar:** both supported.
- **Assessment:** appears correct under action-scoped conflict semantics.
- **Review status:** second review required.

### Unsupported cases

#### `unsupported-01`

- **Input:** `Maybe administrators should delete user accounts.`
- **Attributes/FOL/query/solver/conflict:** not applicable; no structured policy
  should be produced.
- **Expected behavior:** unsupported rejection, not SAT, UNSAT, or UNKNOWN.
- **Conflict:** not applicable because parsing should stop before policy
  comparison.
- **Grammar:** unsupported; modal/ambiguous wording is outside the three forms.
- **Assessment:** rejection appears correct.
- **Review status:** supervisor review required for the product decision that
  “maybe/should” must be rejected rather than represented as uncertainty.

#### `unsupported-02`

- **Input:** `Administrators may delete user accounts.`
- **Attributes/FOL/query/solver/conflict:** not applicable.
- **Expected behavior:** unsupported rejection.
- **Conflict:** not applicable.
- **Grammar:** unsupported; `may` is not `can`, `cannot`, or `only ... can`.
- **Assessment:** rejection appears correct under the documented grammar.
- **Review status:** second review required.

#### `unsupported-03`

- **Input:** `If administrators are on call, they can delete user accounts.`
- **Attributes/FOL/query/solver/conflict:** not applicable.
- **Expected behavior:** unsupported rejection; conditional semantics are not
  encoded.
- **Conflict:** not applicable.
- **Grammar:** unsupported.
- **Assessment:** rejection appears correct; requires review if conditions are
  intended for a future grammar extension.
- **Review status:** supervisor review required.

#### `unsupported-04`

- **Input:** `Users with clearance level 3 can export reports.`
- **Attributes/FOL/query/solver/conflict:** not applicable.
- **Expected behavior:** unsupported rejection; attribute conditions are not
  encoded.
- **Conflict:** not applicable.
- **Grammar:** unsupported.
- **Assessment:** rejection appears correct under the current grammar.
- **Review status:** supervisor review required because this is the closest
  case to real ABAC semantics.

#### `unsupported-05`

- **Input:** empty string.
- **Attributes/FOL/query/solver/conflict:** not applicable.
- **Expected behavior:** unsupported rejection for empty input.
- **Conflict:** not applicable.
- **Grammar:** unsupported.
- **Assessment:** rejection is mathematically and operationally appropriate.
- **Review status:** second review required.

### Normalization and additional scope cases

#### `case-01`

- **Input:** `administrators can delete user accounts.`
- **Structured attributes:** `Administrators`, `Delete`, `UserAccounts`, `Allow`,
  `role_permission`, no conditions.
- **FOL:** `∀x (Administrators(x) → CanDelete(x, UserAccounts))`.
- **Query:** entailment; `K = {Administrators(Alice), Administrators(Alice) → CanDelete(Alice, UserAccounts)}`;
  `C = CanDelete(Alice, UserAccounts)`.
- **Expected result:** `UNSAT`.
- **Conflict:** no.
- **Grammar:** supported; capitalization normalization is allowed.
- **Assessment:** appears correct, but only tests case normalization.
- **Review status:** second review required.

#### `case-02`

- **Input:** `Only data stewards can export reports.`
- **Structured attributes:** `DataStewards`, `Export`, `Reports`, `Allow`,
  `only_role`, no conditions.
- **FOL:** `∀x (CanExport(x, Reports) → DataStewards(x))`.
- **Query:** entailment; `K = {CanExport(Alice, Reports), CanExport(Alice, Reports) → DataStewards(Alice)}`;
  `C = DataStewards(Alice)`.
- **Expected result:** `UNSAT`.
- **Conflict:** no.
- **Grammar:** supported.
- **Assessment:** formula direction appears correct; request fact is generated
  from the capability antecedent.
- **Review status:** supervisor review required.

#### `case-03`

- **Input:** `Guests cannot view tickets.`
- **Structured attributes:** `Guests`, `View`, `Tickets`, `Deny`,
  `role_prohibition`, no conditions.
- **FOL:** `∀x (Guests(x) → ¬CanView(x, Tickets))`.
- **Query:** entailment; `K = {Guests(Alice), Guests(Alice) → ¬CanView(Alice, Tickets)}`;
  `C = ¬CanView(Alice, Tickets)`.
- **Expected result:** `UNSAT`.
- **Conflict:** no.
- **Grammar:** supported.
- **Assessment:** appears correct; repeats the prohibition pattern.
- **Review status:** second review required.

#### `case-04`

- **Input:** `Administrators can read user profiles.` and
  `Administrators cannot read audit logs.`
- **Structured attributes:** same role with `Read/UserProfiles/Allow` and
  `Read/AuditLogs/Deny`.
- **FOL formulas:** `∀x (Administrators(x) → CanRead(x, UserProfiles))` and
  `∀x (Administrators(x) → ¬CanRead(x, AuditLogs))`.
- **Query:** first-policy entailment; `K = {Administrators(Alice), Administrators(Alice) → CanRead(Alice, UserProfiles)}`;
  `C = CanRead(Alice, UserProfiles)`.
- **Expected result:** `UNSAT`.
- **Conflict:** no; resources differ.
- **Grammar:** both supported.
- **Assessment:** appears correct; the second rule is not used in the stated
  first-policy query.
- **Review status:** second review required.

#### `case-05`

- **Input:** `Support agents can view tickets.` and
  `Support agents cannot delete tickets.`
- **Structured attributes:** same role with `View/Tickets/Allow` and
  `Delete/Tickets/Deny`.
- **FOL formulas:** `∀x (SupportAgents(x) → CanView(x, Tickets))` and
  `∀x (SupportAgents(x) → ¬CanDelete(x, Tickets))`.
- **Query:** first-policy entailment; `K = {SupportAgents(Alice), SupportAgents(Alice) → CanView(Alice, Tickets)}`;
  `C = CanView(Alice, Tickets)`.
- **Expected result:** `UNSAT`.
- **Conflict:** no; actions differ.
- **Grammar:** both supported.
- **Assessment:** appears correct; second policy is relevant only to scope
  classification, not the current entailment query.
- **Review status:** second review required.

#### `case-06`

- **Input:** `Auditors can read audit logs.` and
  `Only administrators can delete user accounts.`
- **Structured attributes:** `Auditors/Read/AuditLogs/Allow/role_permission`
  and `Administrators/Delete/UserAccounts/Allow/only_role`.
- **FOL formulas:** `∀x (Auditors(x) → CanRead(x, AuditLogs))` and
  `∀x (CanDelete(x, UserAccounts) → Administrators(x))`.
- **Query:** first-policy entailment; `K = {Auditors(Alice), Auditors(Alice) → CanRead(Alice, AuditLogs)}`;
  `C = CanRead(Alice, AuditLogs)`.
- **Expected result:** `UNSAT`.
- **Conflict:** no; roles, actions, and resources are different.
- **Grammar:** both supported.
- **Assessment:** appears correct; the second policy is not part of the
  first-policy entailment query.
- **Review status:** second review required.

### Explicit consistency and non-entailment cases

#### `consistency-01`

- **Input:** `Administrators can delete user accounts.` and
  `Administrators cannot delete user accounts.`
- **Structured constraints:** for a fixed subject instance, the set requires
  `CanDelete(Alice, UserAccounts)` and
  `¬CanDelete(Alice, UserAccounts)`. At the universal-rule level, it requires
  both `Administrators(x) → CanDelete(x, UserAccounts)` and
  `Administrators(x) → ¬CanDelete(x, UserAccounts)`.
- **Query:** consistency, not entailment.
- **Expected result:** `UNSAT`; a model cannot satisfy both opposite literals
  for an administrator. If no administrator is assumed, universal rules alone
  could be vacuously satisfiable, so the fixed request individual or explicit
  existence assumption is important.
- **Conflict:** yes; exact scope and opposite effects.
- **Grammar:** both inputs supported.
- **Assessment:** expected `UNSAT` is correct only under the implementation’s
  symbolic encoding or an explicit existence/request-fact assumption. The
  dataset does not itself state that assumption, so this case needs supervisor
  review.
- **Review status:** supervisor review required.

#### `nonentail-01`

- **Input:** `Administrators can read user profiles.`
- **Structured attributes:** `Administrators`, `Read`, `UserProfiles`, `Allow`,
  `role_permission`, no conditions.
- **Independent query:** premises `Administrators(Alice)`; conclusion
  `CanDelete(Alice, UserAccounts)`.
- **Query:** entailment, checking `Administrators(Alice) ∧
  ¬CanDelete(Alice, UserAccounts)`.
- **Expected result:** `SAT`; countermodel: Alice is an administrator who can
  read profiles but cannot delete user accounts. The policy imposes no delete
  permission.
- **Conflict:** no; only a read allow policy is supplied.
- **Grammar:** policy input supported; query is already formal.
- **Assessment:** expected `SAT` is mathematically correct and is a useful
  non-entailment case. It does not test a natural-language request parser.
- **Review status:** supervisor review required.

#### `nonentail-02`

- **Input:** `Administrators can read user profiles.`
- **Structured attributes:** `Administrators`, `Read`, `UserProfiles`, `Allow`,
  `role_permission`, no conditions.
- **Independent query:** premises `Auditors(Alice)`; conclusion
  `CanRead(Alice, UserProfiles)`.
- **Query:** entailment, checking `Auditors(Alice) ∧
  ¬CanRead(Alice, UserProfiles)`.
- **Expected result:** `SAT`; countermodel: Alice is an auditor and not an
  administrator, so the administrator read rule does not entail the
  permission.
- **Conflict:** no.
- **Grammar:** policy input supported; query is formal.
- **Assessment:** expected `SAT` is correct if role predicates are not assumed
  equivalent or overlapping. If the domain permits an individual to have both
  roles, the premises still do not entail the permission because no
  administrator fact or role relation is supplied. Useful non-entailment case.
- **Review status:** supervisor review required.

## Cross-case findings

### Cases that appear mathematically correct

All 27 ordinary supported cases (`allow-*`, `deny-*`, `scope-*`, and
`case-*`) have formulas and exact-scope conflict judgments that follow the
documented grammar. The four `only-*` cases also have the correct reversed
implication direction. The five unsupported cases correctly avoid assigning
formal results. The two non-entailment cases have valid SAT countermodels.

### Cases needing revision of interpretation or documentation

No case necessarily requires changing its current label based solely on this
review. However, the following records need their query semantics clarified:

- `conflict-01`, `conflict-02`, `conflict-03`: entailment labels are attached
  to the first policy while conflict labels describe the full policy set.
- `scope-01`, `scope-02`, `scope-03`: the second policy affects conflict
  classification but not the first-policy entailment query.
- `case-04`, `case-05`, `case-06`: same multi-policy issue.
- `consistency-01`: the UNSAT conclusion depends on an existence or fixed
  individual assumption that is not explicit in the record.

### Cases requiring supervisor or second reviewer

All cases require second review before use as independent ground truth because
this report is AI-authored. Priority should be:

1. `consistency-01`, due to the hidden existence assumption.
2. All `only-*` cases, due to reversed implication and generated capability
   facts.
3. All `conflict-*` and multi-policy scope cases, due to mixed query scope.
4. `unsupported-01`, `unsupported-03`, and `unsupported-04`, because rejecting
   modal/conditional/attribute language is a product and grammar decision.
5. `nonentail-01` and `nonentail-02`, because they are the only explicit
   countermodel cases.

## Recommended follow-up

Before changing the dataset, have two reviewers independently annotate the
priority cases with:

1. Structured attributes.
2. FOL formulas.
3. Explicit premises and conclusion.
4. Query type.
5. A model or contradiction proof for SAT/UNSAT.
6. Conflict scope and assumptions.
7. Supported/unsupported decision.

Then adjudicate disagreements and record the assumptions, especially the
existence of `Alice` in consistency queries. A future held-out set should use
independently authored requests and unseen role/action/resource wording rather
than deriving each request fact from the policy antecedent.
