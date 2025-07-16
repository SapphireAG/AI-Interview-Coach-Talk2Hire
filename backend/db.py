# import os
# from dotenv import load_dotenv
# from motor.motor_asyncio import AsyncIOMotorClient
# from beanie import init_beanie

# # Import your document model
# from models import MediaFile

# # Load .env variables
# load_dotenv()
# DATABASE_URL = os.getenv("DATABASE_URL")

# if not DATABASE_URL:
#     raise ValueError("DATABASE_URL is not set in the .env file.")

# # Create MongoDB client
# client = AsyncIOMotorClient(DATABASE_URL)

# # Get database by name (recommended instead of get_default_database)
# db = client["InterviewCoachDB"]  # replace with your actual DB name if different

# # Initialize Beanie with the document model
# async def init_db():
#     await init_beanie(
#         database=db,
#         document_models=[MediaFile]
#     )
#     print("MongoDB initialized with Beanie")
