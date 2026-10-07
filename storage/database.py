import redis
import json
from pymongo import MongoClient

class PolicyRepository:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017", redis_host: str = "localhost", redis_port: int = 6379):
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client["logishield"]
        self.policies_collection = self.db["verified_policies"]
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)

    def save_verified_policy(self, policy_data: dict) -> bool:
        try:
            self.policies_collection.insert_one(policy_data.copy())
            cache_key = f"policy:{policy_data.get('policy_id')}"
            self.redis_client.set(cache_key, json.dumps(policy_data), ex=3600)
            return True
        except Exception:
            return False