import os
from dotenv import load_dotenv
from pymongo import MongoClient
import ssl
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# MongoDB Connection Configuration
MONGO_URI = os.getenv("MONGO_URI")
MONGO_MIN_POOL_SIZE = int(os.getenv("MONGO_MIN_POOL_SIZE", "5"))
MONGO_MAX_POOL_SIZE = int(os.getenv("MONGO_MAX_POOL_SIZE", "10"))
MONGO_MAX_IDLE_TIME_MS = int(os.getenv("MONGO_MAX_IDLE_TIME_MS", "60000"))
MONGO_TIMEOUT_MS = int(os.getenv("MONGO_TIMEOUT_MS", "5000"))

try:
    # Enhanced MongoDB connection with proper SSL and connection pooling
    client = MongoClient(
        MONGO_URI,
        tls=True,  # Use TLS instead of SSL
        tlsAllowInvalidCertificates=True,  # For development; use proper certificates in production
        minPoolSize=MONGO_MIN_POOL_SIZE,
        maxPoolSize=MONGO_MAX_POOL_SIZE,
        maxIdleTimeMS=MONGO_MAX_IDLE_TIME_MS,
        serverSelectionTimeoutMS=MONGO_TIMEOUT_MS,
        connectTimeoutMS=MONGO_TIMEOUT_MS,
        retryWrites=True,
        w='majority'  # Ensure writes are acknowledged by majority of replicas
    )

    # Test the connection
    client.admin.command('ping')
    logger.info("✅ MongoDB connected successfully!")

    # Initialize database and collections
    db = client["nutritionApp"]  # Database Name
    user_collection = db["users"]  # Collection for user authentication
    user_data_collection = db["userData"]  # Collection for nutrition data

except Exception as e:
    logger.error(f"❌ MongoDB connection failed: {str(e)}")
    raise
