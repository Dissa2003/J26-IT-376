import sys, json
sys.path.insert(0, '.')

from src.policy_verifier.evaluation import evaluate_dataset
from src.policy_verifier.parser import PolicyParser
from src.policy_verifier.pipeline import VerificationPipeline

parser = PolicyParser()
pipeline = VerificationPipeline()

print("=" * 60)
print("DATASET EVALUATION — 67 Labelled Test Cases")
print("=" * 60)

total = 0
passed = 0

# ── v1 & v2: use built-in evaluate_dataset ───────────────────────
for name, path in [
    ('evaluation_dataset.json',    'src/policy_verifier/evaluation_dataset.json'),
    ('evaluation_dataset_v2.json', 'src/policy_verifier/evaluation_dataset_v2.json'),
]:
    result = evaluate_dataset(path)
    if isinstance(result, dict):
        cases = result.get('cases', result.get('results', []))
        t = len(cases)
        p = sum(1 for c in cases if c.get('passed') is True)
    else:
        p = getattr(result, 'passed', 0)
        t = getattr(result, 'total', 0)
    total += t; passed += p
    print(f"  {name}: {p}/{t} passed")

# ── v3: manual evaluation (different schema) ──────────────────────
cases3 = json.load(open('src/policy_verifier/evaluation_dataset_v3.json'))['cases']
p3 = t3 = 0
for c in cases3:
    texts = c.get('policy_texts', [])
    expected = c.get('expected_decision', '').upper()
    if not texts or not expected:
        continue
    t3 += 1
    try:
        pol = parser.parse_text(texts[0])
        r = pipeline.verify(pol)
        status = str(r.solver_result.status).upper()
        ent = r.solver_result.is_entailment
        conflicts = pipeline.conflict_detector.find_conflicts(
            [parser.parse_text(t) for t in texts]
        )
        if len(conflicts) > 0:
            actual = 'DENIED'
        elif 'UNSAT' in status and ent:
            actual = 'ALLOWED'
        else:
            actual = 'NOT_DETERMINED'
        if actual == expected:
            p3 += 1
    except Exception:
        pass
total += t3; passed += p3
print(f"  evaluation_dataset_v3.json: {p3}/{t3} passed")

accuracy = passed / total * 100 if total > 0 else 0
print()
print(f"TOTAL:    {passed}/{total} passed")
print(f"ACCURACY: {accuracy:.1f}%")
print("=" * 60)
