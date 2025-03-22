import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from detect import run

# Path to food calorie dataset
FOOD_DATA_FILE = "E:/Design_Project/food_data/food_calorie_data2.csv"  # Update this with your file path
if not os.path.exists(FOOD_DATA_FILE):
    raise FileNotFoundError(
        f"❌ Food data file not found at {FOOD_DATA_FILE}. Please add it."
    )
print("✅ Food data file loaded successfully! 📄")

# Load food calorie dataset
try:
    food_data = pd.read_csv(FOOD_DATA_FILE)
    print("✅ Food calorie dataset loaded successfully! 📊")
except Exception as e:
    raise Exception(f"❌ Error loading food calorie dataset: {e}")

# Food class mapping
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

        # Scale nutritional values
        info = {
            "Calories": food_row["Avg. Calories per Unit"].values[0] * scaling_factor,
            "Protein (g)": food_row["Protein (g)"].values[0] * scaling_factor,
            "Total Fat (g)": food_row["Total Fat (g)"].values[0] * scaling_factor,
            "Carbohydrates (g)": food_row["Carbohydrates (g)"].values[0]
            * scaling_factor,
        }
        return info
    return None


def annotate_image(image_path, detections):
    """
    Annotates the image with bounding boxes and calorie information.
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for det in detections:
        x, y, w, h = det["bbox"]
        food_name = det["name"]
        calories = det["calories"]
        confidence = det["confidence"]

        # Draw bounding box
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

        # Add label
        label = f"{food_name} ({calories:.2f} kcal, {confidence:.2f})"
        draw.text((x, y - 10), label, fill="red", font=font)

    # Save annotated image
    result_path = "result_image.jpg"
    image.save(result_path)
    print("✅ Annotated image saved successfully! 🖼️")
    return result_path


def main():
    # Input image
    image_path = input("Enter the path to the image: ").strip()
    if not os.path.exists(image_path):
        print(f"❌ Image not found at {image_path}. Please check the path.")
        return
    print("✅ Image loaded successfully! 🖼️")

    # Run YOLO detection
    print("⚙️ Running YOLO detection...")
    weights_path = str("E:/Design_Project/yolov5/models/best.pt")
  # Update with the correct path to your model weights
    if not os.path.exists(weights_path):
        print(f"❌ YOLO model weights not found at {weights_path}.")
        return
    print("✅ YOLO model weights loaded successfully! 🧠")

    try:
        result = run(
            weights=weights_path,  # Absolute path to the weights file
            source=image_path,  # Input image
            save_txt=True,  # Save detection results
            save_conf=True,  # Save confidences
            conf_thres=0.25,  # Confidence threshold
            project="runs/detect",  # Output directory
            name="testapp",  # Subfolder name
            exist_ok=True,  # Overwrite existing folder
        )
        print("✅ YOLO detection completed successfully! 🎯")
    except Exception as e:
        print(f"❌ Error during YOLO detection: {e}")
        return

    # Process detection results
    result_dir = os.path.join("runs", "detect", "testapp")
    labels_dir = os.path.join(result_dir, "labels")
    if not os.path.exists(labels_dir):
        print("❌ No labels found in YOLO output.")
        return
    print("✅ YOLO output labels found and processed. 📂")

    # Parse detected labels
    detections = []
    for label_file in os.listdir(labels_dir):
        with open(os.path.join(labels_dir, label_file), "r") as f:
            for line in f:
                class_id, x, y, w, h, confidence = map(float, line.split())
                class_id = int(class_id)
                food_name = food_class_map.get(class_id, f"Unknown {class_id}")
                detected_weight = 100  # Replace with actual weight estimation logic
                food_info = get_food_info(food_name, detected_weight)

                if food_info:
                    detections.append(
                        {
                            "name": food_name,
                            "bbox": (x, y, w, h),
                            "confidence": confidence,
                            "calories": food_info["Calories"],
                        }
                    )
    print("✅ Detections parsed successfully! 📝")

    # Annotate and display the image
    if detections:
        result_image_path = annotate_image(image_path, detections)
        annotated_image = Image.open(result_image_path)
        annotated_image.show()

        # Print detection info
        print("\n🔍 Detection Results:")
        for det in detections:
            print(
                f"🍴 Food: {det['name']}, Calories: {det['calories']:.2f} kcal, Confidence: {det['confidence']:.2f}"
            )
    else:
        print("❌ No food items detected.")


if __name__ == "__main__":
    main()
