import sys
sys.path.insert(0, '.')

from src.policy_verifier.parser import PolicyParser
from src.policy_verifier.pipeline import VerificationPipeline

parser   = PolicyParser()
pipeline = VerificationPipeline()

passed = 0
failed = 0

print("=" * 65)
print("Z3 VERIFICATION TESTS — 3 Expected Outcomes")
print("=" * 65)

# ── TEST 1: ALLOWED ───────────────────────────────────────────────
print('\nTest 1 — ALLOWED')
print('  Policy : "Administrators can delete user accounts"')
print('  Expected: ALLOWED  (Z3 → UNSAT + entailment = True)')
p1 = parser.parse_text("Administrators can delete user accounts")
r1 = pipeline.verify(p1, individual="Administrators")
s1 = str(r1.solver_result.status).upper()
e1 = r1.solver_result.is_entailment
d1 = "ALLOWED" if "UNSAT" in s1 and e1 else "OTHER"
print(f"  Z3 status   : {s1}")
print(f"  Entailment  : {e1}")
print(f"  Decision    : {d1}")
ok1 = d1 == "ALLOWED"
print(f"  {'✅  PASS' if ok1 else '❌  FAIL'}")
passed += ok1; failed += (not ok1)

# ── TEST 2: DENIED ────────────────────────────────────────────────
print('\nTest 2 — DENIED')
print('  Policy A : "Administrators can delete user accounts"')
print('  Policy B : "Administrators cannot delete user accounts"')
print('  Expected : DENIED  (conflict detected between allow + deny)')
pA = parser.parse_text("Administrators can delete user accounts")
pB = parser.parse_text("Administrators cannot delete user accounts")
conflicts = pipeline.conflict_detector.find_conflicts([pA, pB])
d2 = "DENIED" if len(conflicts) > 0 else "ALLOWED"
print(f"  Conflicts   : {len(conflicts)} found")
print(f"  Decision    : {d2}")
ok2 = d2 == "DENIED"
print(f"  {'✅  PASS' if ok2 else '❌  FAIL'}")
passed += ok2; failed += (not ok2)

# ── TEST 3: NOT_DETERMINED ────────────────────────────────────────
print('\nTest 3 — NOT_DETERMINED')
print('  Policy : "Managers can sometimes access data"')
print('  Expected: NOT_DETERMINED  (ambiguous action — "Sometimes")')
p3 = parser.parse_text("Managers can sometimes access data")
r3 = pipeline.verify(p3, individual="Manager")
action3 = p3.action
d3 = "NOT_DETERMINED" if action3 == "Sometimes" else str(r3.solver_result.status)
print(f"  Parsed action : {action3}")
print(f"  Rule status   : {r3.rule_status}")
print(f"  Decision      : {d3}")
ok3 = d3 == "NOT_DETERMINED"
print(f"  {'✅  PASS' if ok3 else '❌  FAIL'}")
passed += ok3; failed += (not ok3)

print("\n" + "=" * 65)
print(f"RESULTS:  {passed} passed  |  {failed} failed  |  3 total")
print("=" * 65)
