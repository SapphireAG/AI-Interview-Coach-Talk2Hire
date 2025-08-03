from beanie import Document
from pydantic import Field
from typing import Optional
from datetime import datetime, timezone
from bson import ObjectId

class UserAudio(Document):
    username: str = Field(..., description="Username of the uploader")
    file_id: ObjectId = Field(..., description="GridFS file ID")
    transcript: str = Field(..., description="Transcription of the audio file")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when the record was created")

    class Settings:
        name = "media_files"
        indexes = [
            {"fields": ["username"]},
            {"fields": ["created_at"]}
        ]
