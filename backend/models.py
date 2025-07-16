from beanie import Document
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class UserAudio(Document):
    username: str
    audio_path: str
    transcript: str
    

    class Settings:
        name = "media_files"  # MongoDB collection name
