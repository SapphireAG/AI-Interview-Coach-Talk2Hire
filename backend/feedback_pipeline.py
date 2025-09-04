from pymongo import MongoClient
from feedbackllm import run_feedback_pipeline  

# Connecting to MongoDB
MONGO_URI = 'mongodb+srv://singamreddy2024:JyFRA2tFzKHp2aMQ@talk-2-hire.oi4xnnc.mongodb.net/?'
client = MongoClient(MONGO_URI)
db = client["interview_coach"]
questions_col = db["questions"]
results_col = db["feedback_results"]

# Dummy candidate responses
dummy_responses = {
    "TECH001": "Overfitting in ML is when a model fits the training data too well...",
    "TECH002": "Normalization scales numerical features to a standard range...",
    "TECH003": "Entropy in decision trees measures the impurity of a dataset...",
}

# Tone & Emotion (from model or fixed for now)
tone = "confident"
emotion = "neutral"

# Fetch questions
interview_questions = list(questions_col.find().limit(10))

for q in interview_questions:
    qid = q["question_id"]
    question = q["question_text"]
    answer = dummy_responses.get(qid, "This is a sample answer to the question.")

    # Get feedback, confidence, accuracy, tone-emotion score
    feedback_data = run_feedback_pipeline(question, answer, tone, emotion)

    # Store in results collection
    results_col.insert_one({
        "question_id": qid,
        "question": question,
        "answer": answer,
        "feedback": feedback_data["feedback"],
        "ideal_answer": feedback_data["ideal_answer"],
        "confidence": feedback_data["model_confidence"],
        "accuracy": feedback_data["accuracy_score"],
        "tone_emotion_score": feedback_data["tone_emotion_score"],
        "final_score": feedback_data["final_score"]
    })

print("Feedback pipeline completed and stored in DB.")
