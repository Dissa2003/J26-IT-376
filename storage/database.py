import pymongo
from datetime import datetime
from typing import Dict, Any

class VerificationDatabase:
    def __init__(self, db_url: str = "mongodb://localhost:27017/"):
        try:
            self.client = pymongo.MongoClient(db_url, serverSelectionTimeoutMS=2000)
            self.db = self.client["policy_verifier_db"]
            self.logs = self.db["verification_logs"]
        except Exception as e:
            print(f"Database connection warning: {e}")
            self.client = None

    def save_log(self, report: Dict[str, Any]):
        if self.client:
            try:
                log_data = report.copy()
                log_data["timestamp"] = datetime.utcnow()
                self.logs.insert_one(log_data)
            except Exception as e:
                print(f"Failed to save log to MongoDB: {e}")