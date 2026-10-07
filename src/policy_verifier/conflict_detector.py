from z3 import Solver, Bool, Not, sat
from typing import List, Tuple

class ConflictDetector:
    def check_conflicts(self, fol_expressions: List[str]) -> Tuple[str, List[str]]:
        s = Solver()
        variables = {}
        conflicts = []

        for expr in fol_expressions:
            clean_expr = expr.replace("(", "_").replace(")", "").replace(", ", "_")
            
            if "Deny_" in clean_expr:
                base_key = clean_expr.replace("Deny_", "")
                allow_var_name = f"Allow_{base_key}"
                
                v_allow = variables.setdefault(allow_var_name, Bool(allow_var_name))
                s.add(Not(v_allow))
                conflicts.append(expr)
            else:
                v = variables.setdefault(clean_expr, Bool(clean_expr))
                s.add(v)

        result = s.check()
        status = "SAT" if result == sat else "UNSAT"
        return status, (conflicts if status == "UNSAT" else [])