from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import json, os
from rich import print
from dotenv import load_dotenv
load_dotenv()

user_name=os.getenv("MONGODB_USERNAME")
password=os.getenv("MONGDB_PASSWORD")

uri = f"mongodb+srv://{user_name}:{password}@cluster0.0h1sfbi.mongodb.net/funny_hunter"

client = MongoClient(uri)
db = client["funny_hunter"]
notes = db["notes"]



def json_serializer(obj):
    if isinstance(obj, ObjectId):
        return str(obj)

    if isinstance(obj, datetime):
        return obj.isoformat()

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


records = list(notes.find().limit(10))

with open("notes_export.json", "w", encoding="utf-8") as f:
    json.dump(
        records,
        f,
        ensure_ascii=False,
        indent=2,
        default=json_serializer
    )

print(f"Saved {len(records)} records to notes_export.json")