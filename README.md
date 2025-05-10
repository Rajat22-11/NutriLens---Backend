# 🥗 NutriLens: AI-Powered Food Recognition & Nutrition Estimator

## 🚀 Project Overview

NutriLens is a sophisticated web application that combines computer vision and AI to provide instant nutritional analysis of food images. Using a dual-model approach with YOLOv5 for specialized Indian food detection and Google's Gemini Vision API as a fallback, the system offers accurate food recognition and detailed nutritional insights.

### 📸 Sample Screenshots
![Screenshot 2025-03-22 133253](https://github.com/user-attachments/assets/dd6386b3-feb7-43a5-b9ee-c56a401427da)

![Screenshot 2025-03-22 133108](https://github.com/user-attachments/assets/ac3fc746-e5c7-4236-8466-f52bac305309)

![Screenshot 2025-03-22 132931](https://github.com/user-attachments/assets/2d904be1-e5fc-42b4-846a-71396a2ec95f)

![Screenshot 2025-03-25 094828](https://github.com/user-attachments/assets/4efc5b7f-b53a-45c9-b7a8-e83ef059ff7c)



---

## 🎯 Key Features

### 🔍 Dual-Model Food Recognition

- **YOLOv5 Model (Primary)**

  - Specialized in detecting 15 Indian food categories
  - Real-time object detection with portion estimation
  - Offline processing for enhanced privacy
  - Uses bounding box analysis for weight estimation

- **Gemini Vision API (Fallback)**
  - Handles unrecognized or packaged foods
  - Extracts text from food labels and packaging
  - Provides enhanced nutritional insights with tips and facts
  - Offers natural language descriptions of food items

> 🔄 If YOLOv5 fails to detect food items in the image, NutriLens automatically uses the **Gemini Vision API** as a fallback. This ensures that even packaged foods or edge cases without proper bounding boxes are still analyzed effectively using advanced image-to-text processing.

---

## ⚖️ YOLOv5 vs Gemini Vision API – Feature Comparison

| Feature                              | YOLOv5 (Primary Model)                                                       | Gemini Vision API (Fallback)                                                  |
|--------------------------------------|------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| Detection Type                       | Custom object detection using bounding boxes                                | Multimodal image understanding (image-to-text)                                |
| Use Case                             | Regular Indian meals and freshly cooked dishes                              | Packaged food, labels, or failed YOLO detections                              |
| Accuracy on Indian Food              | High (trained on curated Indian dataset)                                    | Generalized, moderate accuracy                                                |
| Portion Estimation                   | Yes – via bounding box area and scaling                                     | No – text-based understanding only                                            |
| Processing Mode                      | Offline – runs locally on server                                             | Online – requires internet and API key                                        |
| Customizability                      | Fully trainable on more classes and data                                    | Limited to Google’s API capabilities                                          |
| Response Enrichment                  | Raw nutritional data from database                                           | Enhanced with emojis, tips, and fun facts                                     |
| Speed                                | Fast inference (~30-50ms per image)                                         | Slower due to API call (~300-800ms latency)                                  |
| Privacy                              | High – no external image transfer                                            | Low – image sent to external service                                          |
| Cost                                 | Free (once model is trained)                                                | API usage may incur charges based on request volume                           |

> 🧠 **Why YOLOv5 is Preferred:**  
YOLOv5 provides real-time inference, local processing, better customization, and accurate portion estimation using visual clues—making it ideal for detecting Indian meals and delivering privacy-focused results.

---

## 📊 Nutritional Analysis

- Comprehensive breakdown of nutrients:
  - Calories, Proteins, Carbohydrates, Fats
  - Fiber, Sugar, Sodium, Cholesterol
- Weight estimation using computer vision
- Personalized health insights and recommendations
- Historical tracking of nutritional intake

---

## 💡 User Experience

- Intuitive React-based interface
- Real-time analysis feedback
- Secure user authentication
- Personal nutrition history tracking
- Interactive data visualizations

---

## 🛠️ Technical Architecture

### Frontend (React PWA)

- **Core Technologies**
  - React with Vite for fast development
  - Material UI for modern interface
  - Recharts for data visualization
  - JWT authentication

### Backend (Flask)

- **Core Components**
  - Flask REST API
  - JWT-based authentication
  - MongoDB for data persistence
  - CORS security configuration

- **AI/ML Pipeline**
  - YOLOv5 for food detection
  - OpenCV for image processing
  - Gemini API integration
  - Pandas for nutritional calculations

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 18+
- MongoDB
- Google Cloud API key (for Gemini)

### Backend Setup

```bash
# Clone repository
git clone <repository-url>
cd NutriLens/backend

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials

# Start the server
python app.py
```

### Frontend Setup

```bash
cd NutriLens/frontend
npm install
npm run dev
```

Visit `http://localhost:5173` to access the application.

---

## 📁 Project Structure

```
NutriLens/
├── backend/
│   ├── models/          # YOLOv5 model weights
│   ├── food_data/       # Nutritional database
│   ├── routes/          # API endpoints
│   └── app.py           # Main application
├── frontend/
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Application pages
│   │   └── services/    # API services
│   └── package.json
├── assets/              # App screenshots for README
└── README.md
```

---

## 🔒 Security Features

- JWT-based authentication
- CORS protection
- Secure credential storage
- Rate limiting on API endpoints
- Input validation and sanitization

---

## 🌟 Future Enhancements

- Support for more food categories
- Enhanced portion estimation
- Meal planning recommendations
- Social sharing features
- Offline mode support
