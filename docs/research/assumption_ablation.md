# Policy verification assumption ablation

This report evaluates the original baseline dataset, the v2 dataset, and their combined 54 cases under four consistency-assumption settings.

| Case ID | Expected | Both on | Witness off | Exclusivity off | Both off |
|---|---:|---:|---:|---:|---:|
| `allow-01` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `allow-02` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `allow-03` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `allow-04` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `allow-05` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `allow-06` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `allow-07` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `allow-08` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `only-01` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `only-02` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `only-03` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `only-04` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `deny-01` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `deny-02` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `deny-03` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `deny-04` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `deny-05` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `conflict-01` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `conflict-02` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `conflict-03` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `scope-01` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `scope-02` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `scope-03` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `unsupported-01` | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| `unsupported-02` | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| `unsupported-03` | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| `unsupported-04` | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| `unsupported-05` | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| `case-01` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `case-02` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `case-03` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `case-04` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `case-05` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `case-06` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `consistency-01` | UNSAT | UNSAT | SAT | UNSAT | SAT |
| `nonentail-01` | SAT | SAT | SAT | SAT | SAT |
| `nonentail-02` | SAT | SAT | SAT | SAT | SAT |
| `v2-consistency-01` | SAT | SAT | SAT | SAT | SAT |
| `v2-consistency-02` | SAT | SAT | SAT | SAT | SAT |
| `v2-consistency-03` | UNSAT | UNSAT | SAT | UNSAT | SAT |
| `v2-consistency-04` | UNSAT | UNSAT | SAT | UNSAT | SAT |
| `v2-consistency-05` | SAT | SAT | SAT | SAT | SAT |
| `v2-consistency-06` | UNSAT | UNSAT | SAT | SAT | SAT |
| `v2-consistency-07` | SAT | SAT | SAT | SAT | SAT |
| `v2-consistency-08` | SAT | SAT | SAT | SAT | SAT |
| `v2-entailment-01` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `v2-entailment-02` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `v2-entailment-03` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `v2-entailment-04` | UNSAT | UNSAT | UNSAT | UNSAT | UNSAT |
| `v2-entailment-05` | SAT | SAT | SAT | SAT | SAT |
| `v2-entailment-06` | SAT | SAT | SAT | SAT | SAT |
| `v2-entailment-07` | SAT | SAT | SAT | SAT | SAT |
| `v2-entailment-08` | SAT | SAT | SAT | SAT | SAT |
| `v2-entailment-09` | SAT | SAT | SAT | SAT | SAT |

## Cases whose result changes

| Case ID | Results by setting | Assumption dependency |
|---|---|---|
| `consistency-01` | both_on=UNSAT, witness_off=SAT, exclusivity_off=UNSAT, both_off=SAT | Result depends on the role-witness assumption. |
| `v2-consistency-03` | both_on=UNSAT, witness_off=SAT, exclusivity_off=UNSAT, both_off=SAT | Result depends on the role-witness assumption. |
| `v2-consistency-04` | both_on=UNSAT, witness_off=SAT, exclusivity_off=UNSAT, both_off=SAT | Result depends on the role-witness assumption. |
| `v2-consistency-06` | both_on=UNSAT, witness_off=SAT, exclusivity_off=SAT, both_off=SAT | Result depends on both assumptions; disabling either one changes the result. |

No labels were modified. `UNSUPPORTED` means the original dataset case was explicitly marked unsupported.
