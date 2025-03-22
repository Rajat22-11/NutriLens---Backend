from datetime import datetime
from bson import ObjectId

class UserData:
    """User data model for MongoDB Atlas - stores food analysis history and nutrition goals"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.analysis_history = []
        self.nutrition_goals = {
            "calories": 2000,
            "protein": 80,
            "carbs": 250,
            "fat": 70,
            "fiber": 25,
            "sugar": 30,
            "sodium": 2000,
            "cholesterol": 300
        }
        self.last_updated = datetime.utcnow()
    
    def add_analysis(self, analysis_data):
        """Add a new food analysis entry to user history"""
        # Check if this is an annotated image (which would be a duplicate)
        image_filename = analysis_data.get("imageFilename", "")
        is_annotated = "annotated" in image_filename
        total_nutrients = analysis_data.get("totalNutrients", {
            "calories": 0,
            "protein": 0,
            "carbs": 0,
            "fat": 0,
            "fiber": 0
        })
        
        # If this is an annotated image and has no nutrition data, skip storing it
        if is_annotated and (not total_nutrients or not total_nutrients.get("calories", 0)):
            print(f"⚠️ Skipping storage of annotated image without nutrition data: {image_filename}")
            return None
            
        analysis_entry = {
            "timestamp": datetime.utcnow(),
            "imageFilename": image_filename,
            "imageBase64": analysis_data.get("imageBase64", ""),
            "detectionSource": analysis_data.get("detectionSource", "manual"),
            "detectedFoods": analysis_data.get("detectedFoods", []),
            "totalNutrients": total_nutrients,
            "healthInsight": analysis_data.get("healthInsight", ""),
            "healthierOptions": analysis_data.get("healthierOptions", []),
            "funFact": analysis_data.get("funFact", "")
        }
        self.analysis_history.append(analysis_entry)
        self.last_updated = datetime.utcnow()
        return analysis_entry
    
    def update_nutrition_goals(self, goals_data):
        """Update user's nutrition goals"""
        self.nutrition_goals = {
            "calories": goals_data.get("calories", 2000),
            "protein": goals_data.get("protein", 80),
            "carbs": goals_data.get("carbs", 250),
            "fat": goals_data.get("fat", 70),
            "fiber": goals_data.get("fiber", 25),
            "sugar": goals_data.get("sugar", 30),
            "sodium": goals_data.get("sodium", 2000),
            "cholesterol": goals_data.get("cholesterol", 300)
        }
        self.last_updated = datetime.utcnow()
    
    def get_daily_totals(self, date=None):
        """Get total nutrients consumed on a specific date"""
        if date is None:
            date = datetime.utcnow().date()
        
        # Filter entries by date
        day_entries = [entry for entry in self.analysis_history 
                     if entry["timestamp"].date() == date]
        
        # Calculate totals
        totals = {
            "calories": 0,
            "protein": 0,
            "carbs": 0,
            "fat": 0,
            "fiber": 0,
            "sugar": 0,
            "sodium": 0,
            "cholesterol": 0
        }
        
        for entry in day_entries:
            nutrients = entry.get("totalNutrients", {})
            for key in totals:
                totals[key] += nutrients.get(key, 0)
        
        return totals
    
    def get_progress_percentage(self, date=None):
        """Calculate percentage of daily goals reached"""
        totals = self.get_daily_totals(date)
        progress = {}
        
        for nutrient, amount in totals.items():
            if nutrient in self.nutrition_goals and self.nutrition_goals[nutrient] > 0:
                progress[nutrient] = min(100, round((amount / self.nutrition_goals[nutrient]) * 100, 1))
            else:
                progress[nutrient] = 0
                
        return progress
    
    def to_dict(self):
        """Convert user data object to dictionary for MongoDB storage"""
        return {
            "userId": self.user_id,
            "analysisHistory": self.analysis_history,
            "nutritionGoals": self.nutrition_goals,
            "lastUpdated": self.last_updated
        }
    
    @staticmethod
    def from_dict(data):
        """Create user data object from MongoDB document"""
        user_data = UserData(data.get("userId"))
        user_data.analysis_history = data.get("analysisHistory", [])
        user_data.nutrition_goals = data.get("nutritionGoals", {
            "calories": 2000,
            "protein": 80,
            "carbs": 250,
            "fat": 70,
            "fiber": 25,
            "sugar": 30,
            "sodium": 2000,
            "cholesterol": 300
        })
        user_data.last_updated = data.get("lastUpdated", datetime.utcnow())
        return user_data