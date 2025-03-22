from flask import jsonify
from bson import ObjectId
from datetime import datetime
from models.user_data_model import UserData
from config.config import user_data_collection  # Updated import

def get_user_data(user_id):
    """Get user data from MongoDB"""
    try:
        user_data_doc = user_data_collection.find_one({"userId": user_id})
        if not user_data_doc:
            # Create new user data if it doesn't exist
            user_data = UserData(user_id)
            user_data_collection.insert_one(user_data.to_dict())
            return user_data
        
        return UserData.from_dict(user_data_doc)
    except Exception as e:
        print(f"Error getting user data: {str(e)}")
        return None

def save_analysis(user_id, analysis_data):
    """Save food analysis to user history"""
    try:
        user_data = get_user_data(user_id)
        if not user_data:
            return jsonify({"error": "Failed to get user data"}), 500
        
        # Add analysis and check if it was skipped (returns None for annotated images without data)
        analysis_entry = user_data.add_analysis(analysis_data)
        
        # If analysis_entry is None, it means it was an annotated image without nutrition data
        if analysis_entry is None:
            return jsonify({"message": "Skipped saving annotated image without nutrition data"}), 200
        
        # Update in database
        user_data_collection.update_one(
            {"userId": user_id},
            {"$set": {
                "analysisHistory": user_data.analysis_history,
                "lastUpdated": datetime.utcnow()
            }}
        )
        
        return jsonify({"message": "Analysis saved successfully"}), 200
    except Exception as e:
        print(f"Error saving analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500

def get_analysis_history(user_id):
    """Get user's food analysis history"""
    try:
        user_data = get_user_data(user_id)
        if not user_data:
            return jsonify({"error": "Failed to get user data"}), 500
        
        # Sort history by timestamp (newest first)
        history = sorted(
            user_data.analysis_history,
            key=lambda x: x.get("timestamp", datetime.min),
            reverse=True
        )
        
        # Format timestamps for JSON serialization
        for entry in history:
            if "timestamp" in entry and isinstance(entry["timestamp"], datetime):
                entry["timestamp"] = entry["timestamp"].isoformat()
        
        return jsonify(history), 200
    except Exception as e:
        print(f"Error getting analysis history: {str(e)}")
        return jsonify({"error": str(e)}), 500

def update_nutrition_goals(user_id, goals_data):
    """Update user's nutrition goals"""
    try:
        user_data = get_user_data(user_id)
        if not user_data:
            return jsonify({"error": "Failed to get user data"}), 500
        
        user_data.update_nutrition_goals(goals_data)
        
        # Update in database
        user_data_collection.update_one(
            {"userId": user_id},
            {"$set": {
                "nutritionGoals": user_data.nutrition_goals,
                "lastUpdated": datetime.utcnow()
            }}
        )
        
        return jsonify({"message": "Nutrition goals updated successfully"}), 200
    except Exception as e:
        print(f"Error updating nutrition goals: {str(e)}")
        return jsonify({"error": str(e)}), 500

def get_nutrition_goals(user_id):
    """Get user's nutrition goals"""
    try:
        user_data = get_user_data(user_id)
        if not user_data:
            return jsonify({"error": "Failed to get user data"}), 500
        
        return jsonify(user_data.nutrition_goals), 200
    except Exception as e:
        print(f"Error getting nutrition goals: {str(e)}")
        return jsonify({"error": str(e)}), 500