from flask import Blueprint, request, jsonify
from controllers.auth_controller import register_user, login_user
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask_cors import cross_origin
from config.config import user_collection  # Import from config instead of app
from bson import ObjectId

# Fix the variable name - change nsauth_bp to auth_bp
auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/signup", methods=["POST"])
@cross_origin(supports_credentials=True)
def signup():
    return register_user(request.json)

@auth_bp.route("/login", methods=["POST"])
@cross_origin(supports_credentials=True)
def login():
    # Get the response from login_user
    response = login_user(request.json)
    
    # Check if response is a Flask Response object
    from flask import Response
    if isinstance(response, Response):
        # Don't add the header here - it's already being added by the @cross_origin decorator
        # response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        return response
    # If it's a tuple with (data, status_code)
    elif isinstance(response, tuple) and len(response) == 2:
        response_data, status_code = response
        # Make sure response_data is a dict, not another Response object
        if isinstance(response_data, Response):
            # Don't add the header here either
            # response_data.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
            return response_data
        # Otherwise create a new response
        resp = jsonify(response_data)
        resp.status_code = status_code
        # Don't add the header here either
        # resp.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        return resp
    
    # Fallback case - just return the response as is
    return response

@auth_bp.route("/profile", methods=["GET", "OPTIONS"])
@cross_origin(origins=["http://localhost:5173"], 
              methods=["GET", "OPTIONS"], 
              allow_headers=["Content-Type", "Authorization"],
              supports_credentials=True)
@jwt_required()
def get_user_profile():
    # Remove duplicate OPTIONS check - keep only one
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
        
    # Get user ID from JWT token
    current_user_id = get_jwt_identity()
    
    if not current_user_id:
        return jsonify({"message": "Unauthorized"}), 401
    
    # No need to import from app anymore
    # Find user in database
    user = user_collection.find_one({"_id": ObjectId(current_user_id)})
    
    if not user:
        return jsonify({"message": "User not found"}), 404
    
    # Return user data without sensitive information
    user_data = {
        "id": str(user["_id"]),
        "name": user.get("name", ""),
        "email": user.get("email", ""),
        "age": user.get("age"),
        "height": user.get("height"),
        "weight": user.get("weight"),
        "customerType": user.get("customerType", "Basic")
    }
    
    return jsonify(user_data)
