from fastapi import FastAPI
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
 
app = FastAPI()

# Allow Flutter frontend to access FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
MONGO_URI= "mongodb+srv://singamreddy2024:JyFRA2tFzKHp2aMQ@talk-2-hire.oi4xnnc.mongodb.net/?"
client = MongoClient(MONGO_URI)
db = client["interview_coach"]
collection = db["questions"]

class QuestionRequest(BaseModel):
    question_type: str
    count: int

@app.post("/get-questions/")
def get_questions(data: QuestionRequest):
    questions = list(collection.find(
        {"question_type": data.question_type},
        {"_id": 0}
    ).limit(data.count))
    return {"questions": questions}
