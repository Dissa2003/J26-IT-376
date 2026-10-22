import os

# Keep the policy service's Redis decision cache out of the test run: a cache
# shared across runs would return earlier results instead of exercising Z3.
os.environ.setdefault("POLICY_REQUEST_CACHE", "0")
