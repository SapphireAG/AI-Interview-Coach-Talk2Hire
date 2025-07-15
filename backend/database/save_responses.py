from pymongo import MongoClient
import datetime

# Replace <username>:<password> with your actual MongoDB Atlas credentials
client = MongoClient("mongodb+srv://singamreddy2024:JyFRA2tFzKHp2aMQ@talk-2-hire.oi4xnnc.mongodb.net/?")
db = client["interview_coach"]
responses_col = db["responses"]

def save_response_to_db(response_data: dict):
    response_data["timestamp"] = datetime.datetime.utcnow()
    result = responses_col.insert_one(response_data)
    return str(result.inserted_id)
