"""
Chemelex — MongoDB Setup Script
Seeds the default admin user and ensures indexes are in place.
Run once: python setup_db.py
"""

import os
from datetime import datetime
from pymongo import MongoClient
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

def setup():
    client = MongoClient(MONGO_URI)
    db = client["chemlex"]

    # Ensure indexes
    db.users.create_index("username", unique=True)
    db.conversations.create_index("user_id")
    db.conversations.create_index("updated_at")

    # Seed admin user
    if db.users.count_documents({"username": "admin"}) == 0:
        db.users.insert_one({
            "username": "admin",
            "password_hash": generate_password_hash("chemlex2024"),
            "full_name": "Administrator",
            "role": "admin",
            "created_at": datetime.utcnow(),
        })
        print("[OK] Admin user created (username: admin, password: chemlex2024)")
    else:
        print("[OK] Admin user already exists")

    # Seed a demo user
    if db.users.count_documents({"username": "demo"}) == 0:
        db.users.insert_one({
            "username": "demo",
            "password_hash": generate_password_hash("demo1234"),
            "full_name": "Demo User",
            "role": "viewer",
            "created_at": datetime.utcnow(),
        })
        print("[OK] Demo user created (username: demo, password: demo1234)")
    else:
        print("[OK] Demo user already exists")

    print(f"\n[OK] MongoDB setup complete at {MONGO_URI}")
    print(f"     Database: chemlex")
    print(f"     Collections: users, conversations")
    client.close()

if __name__ == "__main__":
    setup()
