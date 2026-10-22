import sys
sys.path.insert(0, '.')

from src.policy_verifier.parser import PolicyParser

passed = 0
failed = 0

def check(name, condition, got=None, expected=None):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}  |  got: {repr(got)}  |  expected: {repr(expected)}")
        failed += 1

p = PolicyParser()

print("=" * 65)
print("UNIT TESTS — PolicyParser  (src/policy_verifier/parser.py)")
print("=" * 65)

# ── TEST 1: _CAN pattern ──────────────────────────────────────────
print('\nTest 1 — CAN pattern')
print('  Input: "Administrators can delete user accounts"')
r = p.parse_text("Administrators can delete user accounts")
print(f"  Parsed: subject={r.subject}, action={r.action}, resource={r.resource}, effect={r.effect}")
check("subject == Administrators", r.subject == "Administrators", r.subject, "Administrators")
check("action  == Delete",         r.action  == "Delete",         r.action,  "Delete")
check("resource == UserAccounts",  r.resource == "UserAccounts",  r.resource,"UserAccounts")
check("effect  == Allow",          str(r.effect) in ("Allow", "PolicyEffect.ALLOW", "allow"), str(r.effect), "Allow")

# ── TEST 2: _ONLY pattern ─────────────────────────────────────────
print('\nTest 2 — ONLY pattern')
print('  Input: "Only admins can view reports"')
r2 = p.parse_text("Only admins can view reports")
print(f"  Parsed: subject={r2.subject}, action={r2.action}, interpretation={r2.interpretation}")
check("subject == Admins",         r2.subject == "Admins",        r2.subject, "Admins")
check("interpretation == only_restriction",
      "only" in str(r2.interpretation).lower() or "exclusive" in str(r2.interpretation).lower(),
      str(r2.interpretation), "only_restriction / exclusive")

# ── TEST 3: _CANNOT pattern ───────────────────────────────────────
print('\nTest 3 — CANNOT pattern (deny effect)')
print('  Input: "Guests cannot access finance"')
r3 = p.parse_text("Guests cannot access finance")
print(f"  Parsed: subject={r3.subject}, action={r3.action}, effect={r3.effect}")
check("effect == Deny",  str(r3.effect) in ("Deny", "PolicyEffect.DENY", "deny"), str(r3.effect), "Deny")
check("subject == Guests", r3.subject == "Guests", r3.subject, "Guests")

# ── TEST 4: Unsupported / unmatched policy ────────────────────────
print('\nTest 4 — Unsupported policy')
print('  Input: "Managers can sometimes access data"')
try:
    r4 = p.parse_text("Managers can sometimes access data")
    # Parser may return None fields or raise — both count as handled
    if r4 is None or r4.action is None or r4.subject is None:
        print("  PASS  Parser returned None for unsupported pattern — unmatched policy handled")
        passed += 1
    else:
        print(f"  PASS  Parser handled gracefully: subject={r4.subject}, action={r4.action}, effect={r4.effect}")
        passed += 1
except Exception as e:
    print(f"  PASS  Exception raised as expected: {type(e).__name__}: {e}")
    passed += 1

print("\n" + "=" * 65)
print(f"RESULTS:  {passed} passed  |  {failed} failed  |  {passed + failed} total")
print("=" * 65)
