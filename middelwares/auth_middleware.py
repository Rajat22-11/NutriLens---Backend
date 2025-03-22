from flask import request, jsonify
import jwt
import os
from functools import wraps
from bson import ObjectId

JWT_SECRET = os.getenv("JWT_SECRET")


def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "Token missing!"}), 401
            
        # Remove 'Bearer ' prefix if present
        if token.startswith('Bearer '):
            token = token[7:]

        try:
            # Decode the token and extract user_id
            decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            # Add user_id to kwargs so the decorated function can access it
            kwargs['user_id'] = decoded.get('sub')  # 'sub' is where JWT stores the identity
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired!"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token!"}), 401

    return decorated
