from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms 
from PIL import Image
from pydantic import BaseModel
import torch
import shutil 
import os
from FER import ResEmoteNet
from TranscriptionModel.transcript import transcribe_audio
from feedbackModel.feedback_llm import run_feedback_pipeline
from fastapi import Form
from user_database.db import init_db
import asyncio

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await init_db()

#llm feedback struct
class FeedbackRequest(BaseModel):
    question: str
    answer: str
    tone: str
    emotion: str


# Add CORS middleware
origins = [
    "*"  # Allow all origins
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResEmoteNet(num_classes=7).to(device)
checkpoint=torch.load(r"D:\Talk2Hire\AI-Interview-Coach-Talk2Hire\backend\FER\best_resemotenet_model.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/username/")
async def upload_username(username: str = Form(...)):
    # Optional: Save username to DB if you have a user collection
    return {"message": f"Username '{username}' received."}

@app.post("/predict-emotion")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            predicted = torch.argmax(outputs, dim=1).item()

        return {"emotion": int(predicted)}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



UPLOAD_DIR = "Uploaded_Audio_Files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        transcript = transcribe_audio(file_location)

        os.makedirs("Transcripts", exist_ok=True)
        transcript_path = os.path.join("Transcripts", f"{file.filename}.txt")
        with open(transcript_path, "w") as f:
            f.write(transcript)

        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "transcript": transcript
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    
#  LLM Feedback
@app.post("/generate-feedback/")
async def generate_feedback_api(payload: FeedbackRequest):
    try:
        result = run_feedback_pipeline(
            question=payload.question,
            answer=payload.answer,
            tone=payload.tone,
            emotion=payload.emotion
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Interview Coach API"}



