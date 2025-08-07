import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

# Import your document model(s)
# from models import UserAudio
from models import UserAudio  # <-- ADDED: new model to store username, audio path, and transcript

# Load .env variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the .env file.")

# Create MongoDB client
client = AsyncIOMotorClient(DATABASE_URL)

# Get database by name
db = client["InterviewCoachDB"]  # replace with your actual DB name if needed

from motor.motor_asyncio import AsyncIOMotorGridFSBucket
bucket = AsyncIOMotorGridFSBucket(db)

# Initialize Beanie with document models
async def init_db():
    await init_beanie(
        database=db,
        document_models=[UserAudio]  # <-- MODIFIED: include new UserAudio model
    )
    print("MongoDB initialized with Beanie")
