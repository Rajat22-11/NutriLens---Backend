from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask_cors import cross_origin
from config.config import user_collection
from bson import ObjectId

user_bp = Blueprint("user", __name__)

@user_bp.route("/profile", methods=["GET", "OPTIONS"])
@cross_origin(supports_credentials=True)
@jwt_required()
def get_user_profile():
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return "", 200
        
    try:
        # Get user ID from JWT token
        current_user_id = get_jwt_identity()
        
        # Find user in database
        user = user_collection.find_one({"_id": ObjectId(current_user_id)})
        
        if not user:
            return jsonify({"message": "User not found"}), 404
        
        # Return user data without sensitive information
        user_data = {
            "_id": str(user["_id"]),
            "name": user.get("name", ""),
            "email": user.get("email", ""),
            "age": user.get("age"),
            "height": user.get("height"),
            "weight": user.get("weight"),
            "customerType": user.get("customerType", "Basic")
        }
        
        return jsonify(user_data), 200
        
    except Exception as e:
        print(f"Error fetching user profile: {str(e)}")
        return jsonify({"message": "An error occurred while fetching user profile"}), 500