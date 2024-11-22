from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import cv2

# Load the trained model
model = tf.keras.models.load_model('Apple_disease_resnet50.h5')

# Define apple leaf disease classes
disease_classes = {
    0: 'Healthy',
    1: 'Apple Scab',
    2: 'Black Rot',
    3: 'Cedar Apple Rust'
}

# Disease descriptions, precautions, and treatment information
disease_info = {
    'Healthy': {
        'description': 'The apple leaf is healthy. No issues detected.',
        'precautions': 'Continue regular care and monitoring.',
        'treatment': 'None required.'
    },
    'Apple Scab': {
        'description': 'Apple scab is a fungal disease caused by Venturia inaequalis.',
        'precautions': 'Avoid overhead watering and ensure good airflow.',
        'treatment': 'Apply fungicides such as captan or mancozeb.'
    },
    'Black Rot': {
        'description': 'Black rot is caused by the fungus Botryosphaeria obtusa.',
        'precautions': 'Remove and destroy infected leaves and fruit.',
        'treatment': 'Use fungicides such as copper or captan.'
    },
    'Cedar Apple Rust': {
        'description': 'A fungal disease caused by Gymnosporangium juniperi-virginianae.',
        'precautions': 'Remove nearby cedar trees if possible.',
        'treatment': 'Use fungicides containing myclobutanil or triadimefon.'
    }
}

# Initialize FastAPI
app = FastAPI()

# Preprocess the image
def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    img = np.array(image)
    img_resized = cv2.resize(img, (224, 224))  # Resize to model input size
    img_resized = img_resized / 255.0  # Normalize
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    return img_resized

# Prediction route
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_data = await file.read()
        
        # Preprocess the image for model prediction
        preprocessed_image = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        disease_name = disease_classes[predicted_class]
        confidence = np.max(prediction) * 100
        
        # Return prediction result as JSON
        return JSONResponse({
            "predicted_class": disease_name,
            "confidence": f"{confidence:.2f}%",
            "description": disease_info[disease_name]["description"],
            "precautions": disease_info[disease_name]["precautions"],
            "treatment": disease_info[disease_name]["treatment"]
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

