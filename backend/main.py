from fastapi import FastAPI, File, UploadFile
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
from TranscriptionModel.transcript import transcribe_audio

from pydantic import BaseModel

app = FastAPI()

UPLOAD_DIR = "Uploaded_Audio_Files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)

        # saving the uploaded audio fil
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # transcribing the audio file
        transcript = transcribe_audio(file_location)

        # saving the transcript to a text file
        transcript_path = os.path.join("Transcripts", f"{file.filename}.txt")
        os.makedirs("Transcripts", exist_ok=True)
        with open(transcript_path, "w") as f:
            f.write(transcript)

        
        return {"message": "File uploaded successfully", "filename": file.filename, "transcript": transcript}
    

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
