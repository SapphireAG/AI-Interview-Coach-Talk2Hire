from beanie import Document
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class MediaFile(Document):
    user_id: str
    filename: str
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    transcript: Optional[str] = None
    

    class Settings:
        name = "media_files"  # MongoDB collection name
