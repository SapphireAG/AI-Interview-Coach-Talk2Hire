import whisper

model = whisper.load_model("small.en")

def transcribe_audio(file_path: str) -> str:
    result = model.transcribe(file_path, fp16=False)
    return result["text"]
 