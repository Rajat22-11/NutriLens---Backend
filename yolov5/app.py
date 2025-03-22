import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
from detect import run  # Ensure your YOLOv5's `detect.py` is accessible

# Load food calorie dataset
FOOD_DATA_FILE = "E:/Design_Project/food_data/food_calorie_data2.csv"  # Replace with the correct path
if not os.path.exists(FOOD_DATA_FILE):
    st.error(f"Food data file not found at {FOOD_DATA_FILE}. Please add it.")
    st.stop()

# Load dataset
food_data = pd.read_csv(FOOD_DATA_FILE)

# Map class IDs to food names
food_class_map = {
    0: "Biryani",
    1: "Chole Bhature",
    2: "Dabeli",
    3: "Dal",
    4: "Dhokla",
    5: "Dosa",
    6: "Jalebi",
    7: "Kathi Roll",
    8: "Kofta",
    9: "Naan",
    10: "Pakora",
    11: "Paneer Tikka",
    12: "Panipuri",
    13: "Pav Bhaji",
    14: "Vadapav",
}


def get_food_info(food_name, detected_weight):
    """
    Get food information from the dataset and scale based on detected weight.
    """
    food_row = food_data[food_data["Food Item"] == food_name]
    if not food_row.empty:
        avg_weight = food_row["Avg. Weight per Unit (g)"].values[0]
        scaling_factor = detected_weight / avg_weight

        info = {
            "Calories": food_row["Avg. Calories per Unit"].values[0] * scaling_factor,
            "Protein (g)": food_row["Protein (g)"].values[0] * scaling_factor,
            "Total Fat (g)": food_row["Total Fat (g)"].values[0] * scaling_factor,
            "Carbohydrates (g)": food_row["Carbohydrates (g)"].values[0]
            * scaling_factor,
            "Fiber (g)": food_row["Fiber (g)"].values[0] * scaling_factor,
            "Sugar (g)": food_row["Sugar (g)"].values[0] * scaling_factor,
            "Sodium (mg)": food_row["Sodium (mg)"].values[0] * scaling_factor,
            "Cholesterol (mg)": food_row["Cholesterol (mg)"].values[0] * scaling_factor,
        }
        return info
    return None


# Streamlit Interface
st.title("Food Nutrition Estimator")
st.write("Upload an image to detect food items and estimate their nutritional values.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save uploaded image for YOLO detection
    image_path = os.path.join("uploaded_image.jpg")
    image.save(image_path)

    # Run YOLO detection
    st.write("Running YOLO detection...")
    result = run(
        weights="models/best.pt",  # Path to your YOLO model weights
        source=image_path,  # Input image path
        save_txt=True,  # Save detection results
        save_conf=True,  # Save confidences
        conf_thres=0.25,  # Confidence threshold (adjust as needed)
        project="runs/detect",  # Save results here
        name="streamlit",  # Subfolder name
        exist_ok=True,  # Overwrite existing folder
    )

    # YOLO output directory
    result_dir = os.path.join("runs", "detect", "streamlit")

    # Locate the annotated image file
    detected_image_files = [
        f for f in os.listdir(result_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    if detected_image_files:
        detected_image_path = os.path.join(result_dir, detected_image_files[0])
        st.image(detected_image_path, caption="Detected Foods", use_column_width=True)

        # Process detection results
        labels_dir = os.path.join(result_dir, "labels")
        if os.path.exists(labels_dir):
            detected_foods = []
            for label_file in os.listdir(labels_dir):
                label_path = os.path.join(labels_dir, label_file)
                with open(label_path, "r") as f:
                    for line in f:
                        class_id, x, y, w, h, confidence = line.split()
                        class_id = int(class_id)  # Convert class_id to integer
                        food_name = food_class_map.get(
                            class_id, f"Unknown Class {class_id}"
                        )
                        detected_foods.append(
                            {
                                "name": food_name,
                                "weight": 100,  # Replace with actual weight estimation logic if applicable
                                "confidence": float(confidence),
                            }
                        )

            # Display results
            st.write("### Analysis Results:")
            for food in detected_foods:
                food_name = food["name"]
                detected_weight = food["weight"]
                confidence = food["confidence"]

                food_info = get_food_info(food_name, detected_weight)
                if food_info:
                    st.write(
                        f"**{food_name}** (Confidence: {confidence:.2f}, Estimated Weight: {detected_weight} g)"
                    )
                    st.write(f"- Calories: {food_info['Calories']:.2f} kcal")
                    st.write(f"- Protein: {food_info['Protein (g)']:.2f} g")
                    st.write(f"- Total Fat: {food_info['Total Fat (g)']:.2f} g")
                    st.write(f"- Carbohydrates: {food_info['Carbohydrates (g)']:.2f} g")
                    st.write(f"- Fiber: {food_info['Fiber (g)']:.2f} g")
                    st.write(f"- Sugar: {food_info['Sugar (g)']:.2f} g")
                    st.write(f"- Sodium: {food_info['Sodium (mg)']:.2f} mg")
                    st.write(f"- Cholesterol: {food_info['Cholesterol (mg)']:.2f} mg")
                else:
                    st.write(f"No data found for {food_name}.")
    else:
        st.write("No detections made in the image. Try adjusting the model or input.")
