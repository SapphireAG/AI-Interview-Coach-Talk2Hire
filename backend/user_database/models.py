from beanie import Document
from pydantic import Field
from typing import Optional
from datetime import datetime, timezone

class UserAudio(Document):
    username: str = Field(..., description="Username of the uploader")
    audio_path: str = Field(..., description="Path to the saved audio file")
    transcript: str = Field(..., description="Transcription of the audio file")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when the record was created")

    class Settings:
        name = "media_files"  # MongoDB collection name
        indexes = [
            {"fields": ["username"]},
            {"fields": ["created_at"]}
        ]