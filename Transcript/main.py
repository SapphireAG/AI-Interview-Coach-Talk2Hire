import whisper

model = whisper.load_model("small.en")
result = model.transcribe(r"D:\Talk2Hire\venv\audio\audio_track_1.mp3",fp16 = False)
print(result["text"])
with open(r"AI-Interview-Coach-Talk2Hire\Transcript\transcript.txt","w") as f:
    f.write(result['text'])
