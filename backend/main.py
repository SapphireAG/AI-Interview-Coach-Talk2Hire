from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms 
from PIL import Image
from pydantic import BaseModel
import torch
import shutil
import os
from FER import ResEmoteNet
from TranscriptionModel.transcript import transcribe_audio  # Assuming your transcription code is modular

app = FastAPI()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResEmoteNet(num_classes=7).to(device)
checkpoint=torch.load("/Users/amangolani/AI-Interview-Coach-Talk2Hire/backend/FER/best_resemotenet_model.pth", map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

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