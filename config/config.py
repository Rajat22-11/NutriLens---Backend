import os
from dotenv import load_dotenv
from pymongo import MongoClient
import ssl

# Load environment variables from .env
load_dotenv()

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI")

# Simple connection with just the URI
client = MongoClient(MONGO_URI, tlsInsecure=True)

db = client["nutritionApp"]  # Database Name
user_collection = db["users"]  # Collection for user authentication
user_data_collection = db["userData"]  # Collection for nutrition data

(
    print("✅ MongoDB connected successfully!")
    if client
    else print("❌ MongoDB connection failed!")
)
