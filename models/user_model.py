from pymongo import ASCENDING
from config.config import db, user_collection

# Define the user schema
user_schema = {
    "name": str,
    "email": str,
    "passwordHash": str,
    "age": float,  # Supports decimals
    "height": float,  # Supports decimals
    "weight": float,  # Supports decimals
    "customerType": str,  # "Basic" or "Premium"
    "createdAt": str,
    "updatedAt": str,
}

# Create unique index on email
user_collection.create_index([("email", ASCENDING)], unique=True)
