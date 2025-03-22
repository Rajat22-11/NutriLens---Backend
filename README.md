# AI Nutrition App Backend

This folder contains all the backend code for the AI Nutrition App. The backend is built using Flask and integrates with YOLOv5 for food detection and Google's Gemini API for nutritional analysis.

## Structure

- `app.py`: Main Flask application
- `config/`: Configuration files for the application
- `controllers/`: Controller logic for handling requests
- `models/`: Data models for the application
- `routes/`: API route definitions
- `middlewares/`: Middleware functions
- `uploads/`: Folder for storing uploaded images
- `results/`: Folder for storing analysis results
- `requirements.txt`: Backend dependencies

## Getting Started

To run the backend:

```bash
pip install -r requirements.txt
python app.py
```

The backend will be available at http://localhost:5000
