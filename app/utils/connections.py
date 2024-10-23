from pymongo import MongoClient
from app.utils.settings import (
    MONGO_URI,
    DATABASE_NAME,
    COLLECTION_NAME
)

def get_database():
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    return db

def get_collection():
    db = get_database()
    collection = db[COLLECTION_NAME]
    return collection