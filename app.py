import os
import sys
import pathlib
import base64
import torch
import cv2
import numpy as np
import pandas as pd
import ssl
import google.generativeai as genai
import json
import re
from datetime import datetime
from bs4 import BeautifulSoup

# Force TLS 1.2 (if needed)
ssl.OPENSSL_VERSION
ssl._create_default_https_context = ssl._create_unverified_context


from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
from PIL import Image, ImageDraw, ImageFont
from bson import ObjectId

# Import Authentication Routes and MongoDB configuration
from routes.auth_routes import auth_bp
from routes.user_routes import user_bp
from routes.health_routes import health_bp
from config.config import client, user_collection, user_data_collection 
# MongoDB Collections
try:
    db = client["nutritionApp"]
    user_collection = db["users"]  # Stores login/signup data
    user_data_collection = db["userData"]  # Stores nutrition data
    
    # Verify database connection
    client.admin.command('ping')
    print("✅ MongoDB connection successful!")
except Exception as e:
    print(f"❌ MongoDB connection failed: {str(e)}")
    sys.exit(1)

# Patch for Windows: Fix PosixPath issue
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

# Load Environment Variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Initialize Flask App (single instance)
app = Flask(__name__)

# Enable CORS for ALL routes to fix the CORS issues
CORS(app, 
     resources={r"/*": {
         "origins": ["http://localhost:5173", "http://127.0.0.1:5173", "https://nutrilens-frontend.onrender.com"],
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"],
         "expose_headers": ["Content-Type", "Authorization"],
         "supports_credentials": True
     }},
     intercept_exceptions=False)

# Add this route to handle OPTIONS requests for any endpoint
# @app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
# @app.route('/<path:path>', methods=['OPTIONS'])
# def handle_options(path):
#     return '', 200

# Add these headers to all responses
# @app.after_request
# def add_cors_headers(response):
#     response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#     response.headers.add('Access-Control-Allow-Credentials', 'true')
#     return response

# Configure JWT
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET")
jwt = JWTManager(app)

# Register Authentication Routes
app.register_blueprint(auth_bp, url_prefix="/api/auth")

# Register User Routes
app.register_blueprint(user_bp, url_prefix="/api/user")

# Register Health Check Routes
app.register_blueprint(health_bp, url_prefix="/api")

# Define Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# YOLOv5 Model Configuration
YOLOV5_PATH = os.path.join(BASE_DIR, "yolov5")
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

# Import YOLOv5 Modules
try:
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes
    from utils.torch_utils import select_device
    from utils.plots import save_one_box

    print("✅ YOLOv5 modules imported successfully!")
except ModuleNotFoundError as e:
    print(f"❌ Error importing YOLOv5 modules: {e}")
    sys.exit(1)

# Select Device (GPU/CPU)
# DEVICE = select_device("cpu")
DEVICE = torch.device("cpu")
print(f"🖥️ Using device: {DEVICE}")

# Load YOLO Model with CUDA optimizations
MODEL_WEIGHTS = os.path.join(BASE_DIR, "models", "best.pt")
# Enable half precision for faster inference if using CUDA
# half = False  # disable half precision since we're using CPU
# model = DetectMultiBackend(MODEL_WEIGHTS, device=DEVICE, dnn=False, fp16=half)

#Optimized for Render Deployment, due to Render's GPU limitations
half = False  # Disable half-precision to reduce memory
model = DetectMultiBackend(MODEL_WEIGHTS, device=DEVICE, dnn=False, fp16=half)


# Clear CUDA cache to ensure clean start
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load Food Dataset
FOOD_DATA_FILE = os.path.join(BASE_DIR, "food_data", "food_calorie_data2.csv")
food_data = pd.read_csv(FOOD_DATA_FILE)

# Food Class Mapping
food_class_map = {
    0: "Biryani 🍛",
    1: "Chole Bhature 🍽️",
    2: "Dabeli 🌮",
    3: "Dal 🥣",
    4: "Dhokla 🍰",
    5: "Dosa 🥞",
    6: "Jalebi 🍯",
    7: "Kathi Roll 🌯",
    8: "Kofta 🍢",
    9: "Naan 🍞",
    10: "Pakora 🍟",
    11: "Paneer Tikka 🍢",
    12: "Panipuri 🥟",
    13: "Pav Bhaji 🍛",
    14: "Vadapav 🍔",
}

# ---- HELPER FUNCTIONS ---- #


def estimate_weight(x1, y1, x2, y2, img_width, img_height):
    """Estimate weight of food item based on bounding box area."""
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img_width * img_height
    return (bbox_area / img_area) * 1000


def preprocess_image(image_path):
    """Preprocess image for YOLO model with CUDA optimization."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    img = torch.from_numpy(img).float().to(DEVICE) / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)
    if DEVICE.type != "cpu" and model.fp16:
        img = img.half()
    return img


def annotate_image(image_path, detections):
    """Draw bounding boxes and labels on detected food items."""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        label = f"{food_class_map.get(int(cls), f'Unknown {int(cls)}')} {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), label, fill="red", font=font)
    annotated_path = os.path.join(RESULT_FOLDER, "annotated_image.jpg")
    img.save(annotated_path)
    return annotated_path


def encode_image_to_base64(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def get_food_info(food_name, detected_weight):
    """Retrieve food nutritional info based on the detected weight."""
    # Look up the food item using the first word (e.g., "Biryani")
    food_row = food_data[food_data["Food Item"] == food_name.split()[0]]
    if not food_row.empty:
        avg_weight = food_row["Avg. Weight per Unit (g)"].values[0]
        scaling_factor = detected_weight / avg_weight
        info = {
            "Calories": round(
                food_row["Avg. Calories per Unit"].values[0] * scaling_factor, 2
            ),
            "Protein (g)": round(food_row["Protein (g)"].values[0] * scaling_factor, 2),
            "Total Fat (g)": round(
                food_row["Total Fat (g)"].values[0] * scaling_factor, 2
            ),
            "Carbohydrates (g)": round(
                food_row["Carbohydrates (g)"].values[0] * scaling_factor, 2
            ),
            "Fiber (g)": round(food_row["Fiber (g)"].values[0] * scaling_factor, 2),
            "Sugar (g)": round(food_row["Sugar (g)"].values[0] * scaling_factor, 2),
            "Sodium (mg)": round(food_row["Sodium (mg)"].values[0] * scaling_factor, 2),
            "Cholesterol (mg)": round(
                food_row["Cholesterol (mg)"].values[0] * scaling_factor, 2
            ),
        }
        return info
    return None


def extract_text_with_gemini(image_path):
    """Extract text from image using Gemini's vision model."""
    model_gen = genai.GenerativeModel("gemini-pro-vision")
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
    response = model_gen.generate_content(
        [
            "Extract any readable text from the image related to food labels, nutrition facts, or ingredients."
        ],
        [img_data],
    )
    return response.text if response.text else ""


def get_gemini_response(food_items_str):
    """Generate a nutrition analysis response using Gemini's flash model."""
    model_gen = genai.GenerativeModel("gemini-1.5-flash")
    print(f"📝 Input to Gemini: {food_items_str[:100]}...")
    input_prompt = """Generate a well-structured and engaging nutrition breakdown for the detected food item using the following HTML format:  

        <h2 class="food-title">[Food Name] [Emoji]</h2>
        <p><strong>Weight:</strong> [Weight in grams] g</p>

        <h3>Nutritional Breakdown</h3>
        <ul>
            <li class="nutrition-item"><strong>Calories:</strong> [Value] kcal</li>
            <li class="nutrition-item"><strong>Protein:</strong> [Value] g</li>
            <li class="nutrition-item"><strong>Total Fat:</strong> [Value] g</li>
            <li class="nutrition-item"><strong>Carbs:</strong> [Value] g</li>
            <li class="nutrition-item"><strong>Fiber:</strong> [Value] g</li>
            <li class="nutrition-item"><strong>Sugar:</strong> [Value] g</li>
            <li class="nutrition-item"><strong>Sodium:</strong> [Value] mg</li>
            <li class="nutrition-item"><strong>Cholesterol:</strong> [Value] mg</li>
        </ul>

        <div class="health-insight">
            <h3>⚡ Health Insight</h3>
            <p>[Brief statement on health benefits or concerns]</p>
        </div>

        <div class="healthier-options">
            <h3>✅ How to Make it Healthier</h3>
            <ul>
                <li>[Suggestion 1]</li>
                <li>[Suggestion 2]</li>
                <li>[More suggestions, if available, in this manner]</li>
            </ul>
        </div>

        <div class="fun-fact">
            <h3>💡 Fun Fact</h3>
            <p>[Interesting fact about this food]</p>
        </div>

        Return only the HTML content without any markdown code blocks or backticks.
        Do not include any text before or after the HTML content.
        
        IMPORTANT: Make sure the nutritional values are consistent with the input data and can be accurately extracted for database storage.
    """
    try:
        response = model_gen.generate_content([input_prompt, food_items_str])
        response_text = response.text
        print(f"📝 Raw Gemini response (first 100 chars): {response_text[:100]}...")
        if response_text.startswith("```html"):
            response_text = response_text.replace("```html", "").replace("```", "")
        return response_text
    except Exception as e:
        print(f"❌ Error generating Gemini response: {str(e)}")
        return f"<div class='error-message'>Error generating analysis: {str(e)}</div>"


def extract_nutrition_data_from_html(html_content):
    """Extract structured nutrition data from Gemini's HTML response."""
    try:
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract food name
        food_title = soup.select_one(".food-title")
        food_name = food_title.text.strip() if food_title else "Unknown Food"

        # Extract weight
        weight_text = soup.select_one("p strong")
        weight = 0
        if weight_text and weight_text.parent:
            weight_match = re.search(r"(\d+(?:\.\d+)?)", weight_text.parent.text)
            if weight_match:
                weight = float(weight_match.group(1))

        # Extract nutritional values
        nutrients = {}
        for li in soup.select(".nutrition-item"):
            text = li.text.strip()
            if ":" in text:
                key, value = text.split(":", 1)
                key = key.strip()
                value_match = re.search(r"(\d+(?:\.\d+)?)", value)
                if value_match:
                    value = float(value_match.group(1))
                    nutrients[key.lower()] = value

        # Extract health insights
        health_insight = ""
        insight_element = soup.select_one(".health-insight p")
        if insight_element:
            health_insight = insight_element.text.strip()

        # Extract healthier options
        healthier_options = []
        for li in soup.select(".healthier-options li"):
            option = li.text.strip()
            if option:
                healthier_options.append(option)

        # Extract fun fact
        fun_fact = ""
        fact_element = soup.select_one(".fun-fact p")
        if fact_element:
            fun_fact = fact_element.text.strip()

        return {
            "food_name": food_name,
            "weight": weight,
            "nutrients": {
                "calories": nutrients.get("calories", 0),
                "protein": nutrients.get("protein", 0),
                "carbs": nutrients.get("carbs", 0),
                "fat": nutrients.get("total fat", 0),
                "fiber": nutrients.get("fiber", 0),
                "sugar": nutrients.get("sugar", 0),
                "sodium": nutrients.get("sodium", 0),
                "cholesterol": nutrients.get("cholesterol", 0),
            },
            "health_insight": health_insight,
            "healthier_options": healthier_options,
            "fun_fact": fun_fact,
        }
    except Exception as e:
        print(f"❌ Error extracting nutrition data: {str(e)}")
        return {
            "food_name": "Unknown Food",
            "weight": 0,
            "nutrients": {
                "calories": 0,
                "protein": 0,
                "carbs": 0,
                "fat": 0,
                "fiber": 0,
                "sugar": 0,
                "sodium": 0,
                "cholesterol": 0,
            },
            "health_insight": "",
            "healthier_options": [],
            "fun_fact": "",
        }

def store_analysis_data(user_id, image_filename, image_base64, detection_source, nutrition_data):
    """Store nutrition analysis data in MongoDB."""
    try:
        # Ensure user_id is a string for consistent handling
        user_id_str = str(user_id) if user_id else None
        
        if not user_id_str:
            print("❌ Error: Invalid user ID provided to store_analysis_data")
            return False
            
        # Check if this is an annotated image (which would be a duplicate)
        is_annotated = "annotated" in image_filename
        
        # If this is an annotated image and has no nutrition data, skip storing it
        if is_annotated and (not nutrition_data or not nutrition_data.get("nutrients", {}).get("calories", 0)):
            print(f"⚠️ Skipping storage of annotated image without nutrition data: {image_filename}")
            return True

        # Prepare the analysis entry
        analysis_entry = {
            "timestamp": datetime.utcnow(),
            "imageFilename": image_filename,
            "imageBase64": image_base64,
            "detectionSource": detection_source,
            "detectedFoods": [
                {
                    "name": nutrition_data["food_name"],
                    "weight": nutrition_data["weight"],
                    "nutrients": nutrition_data["nutrients"],
                }
            ],
            "totalNutrients": nutrition_data["nutrients"],
            "healthInsight": nutrition_data["health_insight"],
            "healthierOptions": nutrition_data["healthier_options"],
            "funFact": nutrition_data["fun_fact"],
        }

        # Store data in userData collection
        result = user_data_collection.update_one(
            {"userId": user_id_str},
            {
                "$push": {"analysisHistory": analysis_entry},
                "$set": {"lastUpdated": datetime.utcnow()},
            },
            upsert=True
        )

        if result.modified_count > 0 or result.upserted_id:
            print(f"✅ Successfully stored analysis data for user: {user_id}")
            return True
        else:
            print(f"⚠️ No changes made for user: {user_id}")
            return False

    except Exception as e:
        print(f"❌ Error storing analysis data: {str(e)}")
        return False


# ---- API ENDPOINTS ---- #
@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    # Get the current user ID (JWT Identity)
    current_user_id = get_jwt_identity()

    # Convert string ID to ObjectId
    try:
        current_user_id = ObjectId(current_user_id)
    except Exception as e:
        print(f"❌ Error converting user ID: {str(e)}")
        return jsonify({"error": "Invalid user ID"}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_filename = file.filename
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)
    file.save(image_path)

    print(f"\n🔍 Processing image: {image_filename}")
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print("❌ Failed to load image. Check if the file is valid.")
        return jsonify({"error": "Failed to load image"}), 400

    img_height, img_width, _ = img_cv.shape
    img = preprocess_image(image_path)
    print(f"📊 Image preprocessed: Shape {img.shape}")
    pred = model(img)
    pred = non_max_suppression(pred, 0.25, 0.45)[0]
    sys.stdout.flush()

    if len(pred) == 0:
        print("❌ YOLO detection failed - No food items detected")
        print("⚙️ Switching to Gemini powered detection...")
        sys.stdout.flush()
        extracted_text = extract_text_with_gemini(image_path)
        if extracted_text.strip():
            print(f"📝 Gemini extracted text from image: {extracted_text[:100]}...")
            print("🤖 GEMINI POWERED DETECTION SUCCESSFUL")
            sys.stdout.flush()
            gemini_output = get_gemini_response(
                f"Extracted Text from Image:\n{extracted_text}"
            )
            print("🧠 Generated nutrition analysis with Gemini based on text")
            sys.stdout.flush()

            # Extract structured data from Gemini's HTML response
            nutrition_data = extract_nutrition_data_from_html(gemini_output)

            # Create thumbnail for storage
            thumbnail_image = Image.open(image_path)
            thumbnail_image.thumbnail((300, 300))
            thumbnail_path = os.path.join(RESULT_FOLDER, f"thumbnail_{image_filename}")
            thumbnail_image.save(thumbnail_path)
            thumbnail_base64 = encode_image_to_base64(thumbnail_path)

            # Store analysis data in MongoDB
            store_analysis_data(
                current_user_id,  # Now an ObjectId
                image_filename,
                thumbnail_base64,
                "gemini",
                nutrition_data,
            )

            return jsonify(
                {
                    "detections": [],
                    "gemini_analysis": gemini_output,
                    "text_detected": extracted_text,
                    "detection_source": "gemini",
                    "nutrition_data": nutrition_data,
                }
            )

        print("❌ Gemini couldn't detect food or readable text")
        sys.stdout.flush()
        return jsonify(
            {
                "detections": [],
                "gemini_analysis": "No food or readable text detected.",
                "detection_source": "none",
            }
        )

    print(f"✅ YOLO DETECTION SUCCESSFUL - Found {len(pred)} food items:")
    sys.stdout.flush()
    results_dict = {}
    for det in pred:
        x1, y1, x2, y2, conf, cls = det.tolist()
        food_name = food_class_map.get(int(cls), f"Unknown Class {int(cls)}")
        detected_weight = estimate_weight(x1, y1, x2, y2, img_width, img_height)
        print(
            f"  • {food_name} (Confidence: {conf:.2f}, Est. Weight: {detected_weight:.2f}g)"
        )
        sys.stdout.flush()
        food_info = get_food_info(food_name, detected_weight)
        if food_info:
            results_dict[food_name] = {
                "confidence": round(conf, 2),
                "detected_weight": round(detected_weight, 2),
                "calories": food_info["Calories"],
                "info": food_info,
            }

    formatted_results = []
    for food, data in results_dict.items():
        food_str = f"{food} (Confidence: {data['confidence']}, Weight: {data['detected_weight']} g)\n"
        food_str += f"Calories: {data['info']['Calories']} kcal\n"
        food_str += f"Protein: {data['info']['Protein (g)']} g\n"
        food_str += f"Fat: {data['info']['Total Fat (g)']} g\n"
        food_str += f"Carbs: {data['info']['Carbohydrates (g)']} g\n"
        food_str += f"Fiber: {data['info']['Fiber (g)']} g\n"
        food_str += f"Sugar: {data['info']['Sugar (g)']} g\n"
        food_str += f"Sodium: {data['info']['Sodium (mg)']} mg\n"
        food_str += f"Cholesterol: {data['info']['Cholesterol (mg)']} mg"
        formatted_results.append(food_str)

    food_items_str = "\n\n".join(formatted_results)
    print(f"🍽️ Food items detected: {list(results_dict.keys())}")
    sys.stdout.flush()

    gemini_output = get_gemini_response(food_items_str)
    print(f"🧠 Gemini output (first 100 chars): {gemini_output[:100]}...")
    sys.stdout.flush()

    # Extract structured data from Gemini's HTML response
    nutrition_data = extract_nutrition_data_from_html(gemini_output)

    annotated_image_path = annotate_image(image_path, pred)
    annotated_image_base64 = encode_image_to_base64(annotated_image_path)

    # Create thumbnail for storage
    thumbnail_image = Image.open(image_path)
    thumbnail_image.thumbnail((300, 300))
    thumbnail_path = os.path.join(RESULT_FOLDER, f"thumbnail_{image_filename}")
    thumbnail_image.save(thumbnail_path)
    thumbnail_base64 = encode_image_to_base64(thumbnail_path)

    # Store analysis data in MongoDB
    store_analysis_data(
        current_user_id,  # Now an ObjectId
        image_filename,
        thumbnail_base64,
        "yolo",
        nutrition_data,
    )

    print("🧠 Generated nutrition analysis with Gemini")
    print("📊 Processing complete\n")
    sys.stdout.flush()

    return jsonify(
        {
            "detections": results_dict,
            "gemini_analysis": gemini_output,
            "annotated_image": annotated_image_base64,
            "detection_source": "yolo",
            "nutrition_data": nutrition_data,
        }
    )


@app.route("/api/analysis/history", methods=["GET"])
@jwt_required()
def get_analysis_history():
    """Get user's analysis history from MongoDB."""
    current_user_id = get_jwt_identity()
    
    # Ensure we have a string user ID
    current_user_id_str = str(current_user_id) if current_user_id else None
    
    if not current_user_id_str:
        return jsonify({"error": "Invalid user ID"}), 401

    try:
        user_doc = user_data_collection.find_one({"userId": current_user_id_str})
        if not user_doc:
            return jsonify([])

        # Format the analysis history for frontend
        analysis_history = []
        if "analysisHistory" in user_doc:
            for entry in user_doc["analysisHistory"]:
                analysis_item = {
                    "id": str(entry.get("_id", "")),
                    "timestamp": entry.get("timestamp", datetime.utcnow()).isoformat(),
                    "imageBase64": entry.get("imageBase64", ""),
                    "detectionSource": entry.get("detectionSource", ""),
                    "totalCalories": entry.get("totalNutrients", {}).get("calories", 0),
                    "totalProtein": entry.get("totalNutrients", {}).get("protein", 0),
                    "totalCarbs": entry.get("totalNutrients", {}).get("carbs", 0),
                    "totalFat": entry.get("totalNutrients", {}).get("fat", 0),
                    "foodItems": [],
                }

                # Add detected food items
                for food in entry.get("detectedFoods", []):
                    food_item = {
                        "name": food.get("name", "Unknown"),
                        "weight": food.get("weight", 0),
                        "nutrients": food.get("nutrients", {}),
                    }
                    analysis_item["foodItems"].append(food_item)

                analysis_history.append(analysis_item)

        return jsonify(analysis_history)

    except Exception as e:
        print(f"❌ Error retrieving analysis history: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/user/nutrition/summary", methods=["GET"])
@jwt_required()
def get_nutrition_summary():
    """Get user's nutrition summary data for charts."""
    current_user_id = get_jwt_identity()
    
    # Ensure we have a string user ID
    current_user_id_str = str(current_user_id) if current_user_id else None
    
    if not current_user_id_str:
        return jsonify({"error": "Invalid user ID"}), 401

    try:
        user_doc = user_data_collection.find_one({"userId": current_user_id_str})
        if (
            not user_doc
            or "analysisHistory" not in user_doc
            or not user_doc["analysisHistory"]
        ):
            return jsonify({"daily": [], "weekly": [], "monthly": []})

        # Get the analysis history
        history = user_doc["analysisHistory"]

        # Get user's nutrition goals
        goals = user_doc.get(
            "nutritionGoals", {"calories": 2000, "protein": 80, "carbs": 250, "fat": 70}
        )

        # Process the history for daily, weekly, and monthly summaries
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)

        # Daily summary (last 7 days)
        daily_data = []
        for i in range(min(7, len(history))):
            entry = history[i]
            entry_date = entry.get("timestamp", datetime.utcnow())

            # Format date as "Mon", "Tue", etc.
            day_name = entry_date.strftime("%a")

            daily_data.append(
                {
                    "name": day_name,
                    "calories": entry.get("totalNutrients", {}).get("calories", 0),
                    "protein": entry.get("totalNutrients", {}).get("protein", 0),
                    "carbs": entry.get("totalNutrients", {}).get("carbs", 0),
                    "fat": entry.get("totalNutrients", {}).get("fat", 0),
                    "target": goals.get("calories", 2000),
                }
            )

        # Reverse to show oldest to newest
        daily_data.reverse()

        # Weekly summary (last 4 weeks)
        weekly_data = []
        # Group by week and calculate averages
        # This is a simplified approach - in production, you'd want to group by actual week numbers
        for i in range(min(4, len(history) // 7 + 1)):
            week_entries = history[i * 7 : i * 7 + 7]
            if week_entries:
                avg_calories = sum(
                    entry.get("totalNutrients", {}).get("calories", 0)
                    for entry in week_entries
                ) / len(week_entries)
                avg_protein = sum(
                    entry.get("totalNutrients", {}).get("protein", 0)
                    for entry in week_entries
                ) / len(week_entries)
                avg_carbs = sum(
                    entry.get("totalNutrients", {}).get("carbs", 0)
                    for entry in week_entries
                ) / len(week_entries)
                avg_fat = sum(
                    entry.get("totalNutrients", {}).get("fat", 0)
                    for entry in week_entries
                ) / len(week_entries)
                weekly_data.append(
                    {
                        "name": f"Week {i+1}",
                        "calories": round(avg_calories, 1),
                        "protein": round(avg_protein, 1),
                        "carbs": round(avg_carbs, 1),
                        "fat": round(avg_fat, 1),
                        "target": goals.get("calories", 2000),
                    }
                )

        # Monthly summary (last 3 months)
        monthly_data = []
        # This is a simplified approach - in production, you'd want to group by actual months
        for i in range(min(3, len(history) // 30 + 1)):
            month_entries = history[i * 30 : i * 30 + 30]
            if month_entries:
                avg_calories = sum(
                    entry.get("totalNutrients", {}).get("calories", 0)
                    for entry in month_entries
                ) / len(month_entries)
                avg_protein = sum(
                    entry.get("totalNutrients", {}).get("protein", 0)
                    for entry in month_entries
                ) / len(month_entries)
                avg_carbs = sum(
                    entry.get("totalNutrients", {}).get("carbs", 0)
                    for entry in month_entries
                ) / len(month_entries)
                avg_fat = sum(
                    entry.get("totalNutrients", {}).get("fat", 0)
                    for entry in month_entries
                ) / len(month_entries)

                # Get the month name from the first entry in the group
                month_name = (
                    month_entries[0].get("timestamp", datetime.utcnow()).strftime("%b")
                )

                monthly_data.append(
                    {
                        "name": month_name,
                        "calories": round(avg_calories, 1),
                        "protein": round(avg_protein, 1),
                        "carbs": round(avg_carbs, 1),
                        "fat": round(avg_fat, 1),
                        "target": goals.get("calories", 2000),
                    }
                )

        # Calculate today's total nutrients for all entries from the current day
        today = datetime.utcnow().date()
        today_entries = [entry for entry in history if entry.get("timestamp", datetime.min).date() == today]
        
        # Initialize total nutrients dictionary
        today_total_nutrients = {
            "calories": 0,
            "protein": 0,
            "carbs": 0,
            "fat": 0,
            "fiber": 0,
            "sugar": 0,
            "sodium": 0,
            "cholesterol": 0
        }
        
        # Sum up all nutrients from today's entries
        for entry in today_entries:
            entry_nutrients = entry.get("totalNutrients", {})
            for nutrient in today_total_nutrients:
                today_total_nutrients[nutrient] += entry_nutrients.get(nutrient, 0)
        
        # Calculate nutrient distributions for pie charts
        nutrient_distribution = []

        total_macros = (
            today_total_nutrients.get("protein", 0)
            + today_total_nutrients.get("carbs", 0)
            + today_total_nutrients.get("fat", 0)
        )

        if total_macros > 0:
            nutrient_distribution = [
                {
                    "name": "Protein",
                    "value": round(
                        today_total_nutrients.get("protein", 0) / total_macros * 100, 1
                    ),
                    "color": "#8884d8",
                },
                {
                    "name": "Carbs",
                    "value": round(
                        today_total_nutrients.get("carbs", 0) / total_macros * 100, 1
                    ),
                    "color": "#82ca9d",
                },
                {
                    "name": "Fat",
                    "value": round(today_total_nutrients.get("fat", 0) / total_macros * 100, 1),
                    "color": "#ffc658",
                },
            ]

        # Calculate progress towards daily goals
        goal_progress = [
            {
                "name": "Calories",
                "current": round(today_total_nutrients.get("calories", 0), 1),
                "goal": goals.get("calories", 2000),
                "unit": "kcal",
                "color": "#ff7300",
            },
            {
                "name": "Protein",
                "current": round(today_total_nutrients.get("protein", 0), 1),
                "goal": goals.get("protein", 80),
                "unit": "g",
                "color": "#8884d8",
            },
            {
                "name": "Carbs",
                "current": round(today_total_nutrients.get("carbs", 0), 1),
                "goal": goals.get("carbs", 250),
                "unit": "g",
                "color": "#82ca9d",
            },
            {
                "name": "Fat",
                "current": round(today_total_nutrients.get("fat", 0), 1),
                "goal": goals.get("fat", 70),
                "unit": "g",
                "color": "#ffc658",
            },
            {
                "name": "Fiber",
                "current": round(today_total_nutrients.get("fiber", 0), 1),
                "goal": goals.get("fiber", 25),
                "unit": "g",
                "color": "#a4de6c",
            },
        ]

        return jsonify(
            {
                "daily": daily_data,
                "weekly": weekly_data,
                "monthly": monthly_data,
                "nutrientDistribution": nutrient_distribution,
                "goalProgress": goal_progress,
            }
        )

    except Exception as e:
        print(f"❌ Error generating nutrition summary: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/user/nutrition/goals", methods=["GET", "POST"])
@jwt_required()
def manage_nutrition_goals():
    """Get or update user's nutrition goals."""
    current_user_id = get_jwt_identity()
    
    # Ensure we have a string user ID
    current_user_id_str = str(current_user_id) if current_user_id else None
    
    if not current_user_id_str:
        return jsonify({"error": "Invalid user ID"}), 401

    # Validate user exists
    user_exists = user_collection.find_one({"_id": ObjectId(current_user_id)})
    if not user_exists:
        return jsonify({"error": "User not found"}), 404

    if request.method == "GET":
        try:
            user_data = user_data_collection.find_one(
                {"userId": current_user_id_str},
                {"nutritionGoals": 1, "_id": 0}
            )
            
            # Return default goals if no custom goals are set
            default_goals = {
                "calories": 2000,
                "protein": 80,
                "carbs": 250,
                "fat": 70,
                "fiber": 25,
                "sugar": 30,
                "sodium": 2000,
                "cholesterol": 300,
            }
            
            return jsonify(user_data.get("nutritionGoals", default_goals) if user_data else default_goals)

        except Exception as e:
            print(f"❌ Error retrieving nutrition goals: {str(e)}")
            return jsonify({"error": "Failed to retrieve nutrition goals"}), 500

    elif request.method == "POST":
        try:
            goals_data = request.json

            # Validate input data
            required_fields = ["calories", "protein", "carbs", "fat"]
            if not all(field in goals_data for field in required_fields):
                return jsonify({"error": "Missing required fields"}), 400

            # Validate numeric values
            for field in goals_data:
                try:
                    goals_data[field] = float(goals_data[field])
                except (ValueError, TypeError):
                    return jsonify({"error": f"Invalid value for {field}"}), 400

            # Update user's nutrition goals with validation
            result = user_data_collection.update_one(
                {"userId": current_user_id_str},
                {
                    "$set": {
                        "nutritionGoals": {
                            "calories": goals_data.get("calories", 2000),
                            "protein": goals_data.get("protein", 80),
                            "carbs": goals_data.get("carbs", 250),
                            "fat": goals_data.get("fat", 70),
                            "fiber": goals_data.get("fiber", 25),
                            "sugar": goals_data.get("sugar", 30),
                            "sodium": goals_data.get("sodium", 2000),
                            "cholesterol": goals_data.get("cholesterol", 300),
                        },
                        "lastUpdated": datetime.utcnow(),
                    }
                },
                upsert=True
            )

            if result.modified_count > 0 or result.upserted_id:
                return jsonify({"message": "Nutrition goals updated successfully"})
            else:
                return jsonify({"message": "No changes made to nutrition goals"})

        except Exception as e:
            print(f"❌ Error updating nutrition goals: {str(e)}")
            return jsonify({"error": "Failed to update nutrition goals"}), 500

@app.route("/api/user/meal-trends", methods=["GET"])
@jwt_required()
def get_meal_trends():
    """Get user's meal trends and patterns."""
    current_user_id = get_jwt_identity()

    try:
        user_doc = user_data_collection.find_one({"userId": current_user_id})
        if (
            not user_doc
            or "analysisHistory" not in user_doc
            or not user_doc["analysisHistory"]
        ):
            return jsonify(
                {"commonFoods": [], "mealTimings": [], "weekdayPatterns": []}
            )

        # Get the analysis history
        history = user_doc["analysisHistory"]

        # Identify common foods
        food_counter = {}
        for entry in history:
            for food in entry.get("detectedFoods", []):
                food_name = food.get("name", "Unknown")
                if food_name in food_counter:
                    food_counter[food_name] += 1
                else:
                    food_counter[food_name] = 1

        # Get top 5 common foods
        common_foods = sorted(
            [{"name": k, "count": v} for k, v in food_counter.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:5]

        # Analyze meal timings
        meal_hours = []
        for entry in history:
            timestamp = entry.get("timestamp")
            if timestamp:
                hour = timestamp.hour
                meal_hours.append(hour)

        # Group meal times
        morning_meals = sum(1 for h in meal_hours if 6 <= h < 11)
        noon_meals = sum(1 for h in meal_hours if 11 <= h < 15)
        evening_meals = sum(1 for h in meal_hours if 15 <= h < 19)
        night_meals = sum(1 for h in meal_hours if h >= 19 or h < 6)

        meal_timings = [
            {"name": "Morning (6-11AM)", "count": morning_meals},
            {"name": "Noon (11AM-3PM)", "count": noon_meals},
            {"name": "Evening (3-7PM)", "count": evening_meals},
            {"name": "Night (7PM-6AM)", "count": night_meals},
        ]

        # Analyze weekday patterns
        weekday_meals = {}
        for entry in history:
            timestamp = entry.get("timestamp")
            if timestamp:
                weekday = timestamp.strftime("%a")  # Mon, Tue, etc.
                if weekday in weekday_meals:
                    weekday_meals[weekday] += 1
                else:
                    weekday_meals[weekday] = 1

        # Format weekday data
        weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekday_patterns = [
            {"name": day, "count": weekday_meals.get(day, 0)} for day in weekday_order
        ]

        return jsonify(
            {
                "commonFoods": common_foods,
                "mealTimings": meal_timings,
                "weekdayPatterns": weekday_patterns,
            }
        )

    except Exception as e:
        print(f"❌ Error generating meal trends: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/auth/profile", methods=["GET", "OPTIONS"])
@cross_origin(supports_credentials=True)
@jwt_required()
def get_auth_profile():
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
        
# Add a new endpoint for saving analysis
@app.route("/api/analysis/save", methods=["POST", "OPTIONS"])
@jwt_required()
def save_analysis():
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return "", 200
        
    try:
        # Get user ID from JWT token
        current_user_id = get_jwt_identity()
        
        # Get data from request
        analysis_data = request.json
        
        if not analysis_data:
            return jsonify({"error": "No data provided"}), 400
            
        # Check if this is an annotated image (which would be a duplicate)
        image_filename = analysis_data.get("imageFilename", "")
        is_annotated = "annotated" in image_filename
        total_nutrients = analysis_data.get("totalNutrients", {})
        
        # If this is an annotated image and has no nutrition data, skip storing it
        if is_annotated and (not total_nutrients or not total_nutrients.get("calories", 0)):
            print(f"⚠️ Skipping storage of annotated image without nutrition data: {image_filename}")
            return jsonify({"message": "Skipped saving annotated image without nutrition data"}), 200
            
        # Prepare the analysis entry
        analysis_entry = {
            "timestamp": datetime.utcnow(),
            "imageFilename": image_filename,
            "imageBase64": analysis_data.get("imageBase64", ""),
            "detectionSource": analysis_data.get("detectionSource", "manual"),
            "detectedFoods": analysis_data.get("detectedFoods", []),
            "totalNutrients": total_nutrients,
            "healthInsight": analysis_data.get("healthInsight", ""),
            "healthierOptions": analysis_data.get("healthierOptions", []),
            "funFact": analysis_data.get("funFact", ""),
        }
        
        # Find the user document
        result = user_data_collection.update_one(
            {"userId": current_user_id},
            {
                "$push": {"analysisHistory": analysis_entry},
                "$set": {"lastUpdated": datetime.utcnow()},
            },
            upsert=True  # Create document if it doesn't exist
        )
        
        if result.modified_count > 0 or result.upserted_id:
            print(f"✅ Successfully stored analysis data for user: {current_user_id}")
            return jsonify({"message": "Analysis saved successfully"}), 200
        else:
            print(f"⚠️ No changes made for user: {current_user_id}")
            return jsonify({"message": "No changes made"}), 200
            
    except Exception as e:
        print(f"❌ Error saving analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=False, port=port)

