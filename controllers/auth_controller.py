from flask import jsonify, request
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
from config.config import user_collection  # Changed from users_collection to user_collection
from flask_jwt_extended import create_access_token
import traceback  # For detailed error logging

# Secret key for JWT (Ensure it is loaded properly)
JWT_SECRET = os.getenv("JWT_SECRET")


def register_user(data):
    try:
        print(f"🔍 Signup Request Received: {data}")  # Log incoming data

        # Validate required fields
        required_fields = ["name", "email", "password", "customerType"]
        for field in required_fields:
            if field not in data or not data[field]:
                print(f"❌ Missing field: {field}")
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Check if email already exists
        existing_user = user_collection.find_one({"email": data["email"]})
        if existing_user:
            print("❌ Email already exists!")
            return jsonify({"error": "Email already exists!"}), 400

        # Hash password securely
        hashed_password = generate_password_hash(
            data["password"], method="pbkdf2:sha256"
        )

        # Create user document
        new_user = {
            "name": data["name"],
            "email": data["email"],
            "passwordHash": hashed_password,
            "age": float(data.get("age", 0)),  # Supports decimals
            "height": float(data.get("height", 0)),  # Supports decimals
            "weight": float(data.get("weight", 0)),  # Supports decimals
            "customerType": data.get("customerType", "Basic"),  # Default: Basic
            "createdAt": datetime.utcnow().isoformat(),
            "updatedAt": datetime.utcnow().isoformat(),
        }

        # Insert user into database
        result = user_collection.insert_one(new_user)
        new_user_id = result.inserted_id
        print("✅ User successfully created!")

        # Return a JWT token on signup with user ID as identity
        access_token = create_access_token(identity=str(new_user_id))
        return (
            jsonify({
                "message": "Account created successfully!", 
                "token": access_token,
                "user": {
                    "_id": str(new_user_id),
                    "name": data.get("name", ""),
                    "email": data.get("email", ""),
                    "customerType": data.get("customerType", "Basic")
                }
            }),
            201,
        )

    except Exception as e:
        print(f"❌ Signup Error: {str(e)}")
        traceback.print_exc()  # Prints detailed error traceback for debugging
        return jsonify({"error": "Internal server error"}), 500


def login_user(data):
    try:
        print(f"🔍 Login Request Received: {data}")  # Log incoming data

        # Validate required fields
        if "email" not in data or "password" not in data:
            print("❌ Missing email or password")
            return jsonify({"error": "Email and password are required"}), 400

        # Find user by email
        user = user_collection.find_one({"email": data["email"]})
        if not user:
            print("❌ User not found")
            return jsonify({"error": "User not found!"}), 404

        # Check password
        if not check_password_hash(user["passwordHash"], data["password"]):
            print("❌ Invalid credentials")
            return jsonify({"error": "Invalid credentials!"}), 401

        # Generate JWT token with user ID as identity
        access_token = create_access_token(identity=str(user["_id"]))
        print("✅ Login successful!")

        # Return user data along with token (including ObjectId as string)
        return jsonify({
            "token": access_token, 
            "message": "Login successful!",
            "user": {
                "_id": str(user["_id"]),
                "name": user.get("name", ""),
                "email": user.get("email", ""),
                "customerType": user.get("customerType", "Basic")
            }
        }), 200

    except Exception as e:
        print(f"❌ Login Error: {str(e)}")
        traceback.print_exc()  # Prints detailed error traceback for debugging
        return jsonify({"error": "Internal server error"}), 500
