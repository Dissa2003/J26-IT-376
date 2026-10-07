import time
import json
from src.policy_verifier.verifier import PolicyVerificationEngine

def evaluate_performance():
    engine = PolicyVerificationEngine()
    
    with open("tests/test_policies/benchmark_suite.json", "r") as f:
        dataset = json.load(f)

    correct_predictions = 0
    total_time_ms = 0.0

    print("=== RUNNING POLICY VERIFICATION ENGINE BENCHMARK ===")
    for case in dataset:
        start = time.perf_counter()
        res = engine.process_and_verify(case["policies"])
        elapsed = (time.perf_counter() - start) * 1000

        total_time_ms += elapsed
        
        # Access Pydantic attributes with dot notation (.status)
        if res.status == case["expected"]:
            correct_predictions += 1

        print(f"Case #{case['id']} | Expected: {case['expected']} | Got: {res.status} | Latency: {elapsed:.3f} ms")

    accuracy = (correct_predictions / len(dataset)) * 100 if dataset else 0.0
    avg_latency = total_time_ms / len(dataset) if dataset else 0.0

    print("\n=== EVALUATION RESULTS ===")
    print(f"Total Cases Benchmarked: {len(dataset)}")
    print(f"Z3 Solver Accuracy:     {accuracy:.2f}%")
    print(f"Average Processing Time: {avg_latency:.3f} ms")

if __name__ == "__main__":
    evaluate_performance()