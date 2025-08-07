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
from user_database.db import init_db , db , bucket
from user_database.models import UserAudio
import asyncio
from motor.motor_asyncio import AsyncIOMotorGridFSBucket

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

# @app.post("/upload-audio/")
# async def upload_audio(file: UploadFile = File(...)):
#     try:
#         file_location = os.path.join(UPLOAD_DIR, file.filename)

#         with open(file_location, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         transcript = transcribe_audio(file_location)

#         os.makedirs("Transcripts", exist_ok=True)
#         transcript_path = os.path.join("Transcripts", f"{file.filename}.txt")
#         with open(transcript_path, "w") as f:
#             f.write(transcript)

#         return {
#             "message": "File uploaded successfully",
#             "filename": file.filename,
#             "transcript": transcript
#         }

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)
    

@app.post("/upload-audio/")
async def upload_audio(username: str = Form(...), file: UploadFile = File(...)):
    """
    Handles uploading an audio file, storing it in GridFS,
    and saving its metadata and transcript to the database.
    """
    file_location = None  # Initialize file_location to ensure it exists for the finally block
    try:
        # --- Step 1: Temporarily save file for transcription ---
        # Some transcription libraries work best with a file path.
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # --- Step 2: Transcribe the audio file ---
        transcript = transcribe_audio(file_location)

        # --- Step 3: Upload the audio file to GridFS ---
        # Rewind the file-like object to the beginning to read its content again
        await file.file.seek(0)
        
        # Use the GridFS bucket to upload the file stream
        file_id = await bucket.upload_from_stream(
            filename=file.filename,
            source=file.file,
            metadata={"contentType": file.content_type, "username": username}
        )

        # --- Step 4: Save metadata to the 'media_files' collection ---
        # Create a new UserAudio document with the GridFS file_id
        user_audio_doc = UserAudio(
            username=username,
            file_id=file_id,
            transcript=transcript
        )
        # Insert the document into the database
        await user_audio_doc.insert()

        return {
            "message": "File uploaded and processed successfully",
            "username": username,
            "filename": file.filename,
            "file_id": str(file_id),  # Convert ObjectId to string for the JSON response
            "transcript": transcript
        }

    except Exception as e:
        # Return a detailed error message if anything goes wrong
        return JSONResponse(content={"error": f"An unexpected error occurred: {str(e)}"}, status_code=500)

    # finally:
    #     # --- Step 5: Clean up the temporary file ---
    #     # Ensure the temporary file is deleted after the request is complete
    #     if file_location and os.path.exists(file_location): 
    #         os.remove(file_location)
        
        # No need to save the transcript to a separate file anymore,
        # as it's now stored in the database.

@app.get("/download-audio/{file_id}")
async def download_audio(file_id: str):
    """
    Downloads an audio file from GridFS by its ID.
    """
    from bson import ObjectId
    from starlette.responses import StreamingResponse

    try:
        grid_out = await bucket.open_download_stream(ObjectId(file_id))
        
        content_type = grid_out.metadata.get("contentType", "application/octet-stream")

        return StreamingResponse(
            grid_out,
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename=\"{grid_out.filename}\""}
        )
    except Exception as e:
        return JSONResponse(content={"error": f"File not found or error streaming: {str(e)}"}, status_code=404)

    
    
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



