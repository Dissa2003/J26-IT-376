import json

# Check how many cases have NO expected_outcome (NONE) and how many have SAT vs UNSAT
# For v1: skip NONE cases, compare only where expected exists
cases1 = json.load(open('src/policy_verifier/evaluation_dataset.json'))['cases']
print("v1 breakdown:")
unsat = sat = none_count = 0
for c in cases1:
    o = c.get('expected', {}).get('solver_outcome', '')
    qt = c.get('expected', {}).get('query_type', '')
    if not o:
        none_count += 1
    elif o == 'UNSAT':
        unsat += 1
    elif o == 'SAT':
        sat += 1
        print(f"  SAT case: id={c['id']} query_type={qt} input={c['input'][:60]}")
print(f"  UNSAT={unsat} SAT={sat} NONE={none_count}")
print(f"  Pipeline always returns UNSAT -> we match {unsat}/{unsat+sat} = {unsat/(unsat+sat)*100:.1f}% on outcome cases")

cases2 = json.load(open('src/policy_verifier/evaluation_dataset_v2.json'))['cases']
print("\nv2 breakdown:")
sat2 = unsat2 = 0
for c in cases2:
    o = c.get('expected_solver_outcome','')
    qt = c.get('query_type','')
    if o == 'SAT':
        sat2 += 1
        print(f"  SAT: id={c['id']} query_type={qt} texts={c.get('policy_texts',[''])[0][:60]}")
    else:
        unsat2 += 1
print(f"  UNSAT={unsat2} SAT={sat2} -> match {unsat2}/{unsat2+sat2} = {unsat2/(unsat2+sat2)*100:.1f}%")

cases3 = json.load(open('src/policy_verifier/evaluation_dataset_v3.json'))['cases']
print("\nv3 breakdown:")
from collections import Counter
ct = Counter(c.get('expected_decision','') for c in cases3)
print(f"  decisions: {dict(ct)}")
for c in cases3:
    if c.get('expected_decision','') != 'ALLOWED':
        print(f"  non-ALLOWED: id={c['id']} expected={c['expected_decision']} texts={c.get('policy_texts',[''])[0][:60]}")
