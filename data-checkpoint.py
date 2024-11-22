import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Load pre-trained models for multiple plant types
models = {
    'Apple': tf.keras.models.load_model('Apple_disease_resnet50.h5'),
    # Add more plant models as needed (e.g., 'Tomato': tf.keras.models.load_model('tomato_disease_model.h5'))
}

disease_classes = {
    
    'Apple': {
        0: 'Healthy',
        1: 'Apple Scab',
        2: 'Black Rot',
        3: 'Cedar Apple Rust'
    }
}

disease_info = {
    'Apple': {
        'Healthy': {
            'description': 'The apple leaf is healthy. No issues detected.',
            'precautions': 'Continue regular care and monitoring.',
            'treatment': 'None required.'
        },
        'Apple Scab': {
            'description': 'A fungal disease caused by Venturia inaequalis.',
            'precautions': 'Avoid overhead watering.',
            'treatment': 'Use fungicides such as captan.'
        },
        'Black Rot': {
            'description': 'A fungal disease caused by Botryosphaeria obtusa.',
            'precautions': 'Prune and destroy infected branches.',
            'treatment': 'Apply fungicides like thiophanate-methyl.'
        },
        'Cedar Apple Rust': {
            'description': 'A fungal disease caused by Gymnosporangium juniperi-virginianae.',
            'precautions': 'Remove nearby cedar trees.',
            'treatment': 'Use fungicides such as myclobutanil.'
        }
    }
}

# Image preprocessing function
def preprocess_image(image):
    img = np.array(image)
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized / 255.0  # Normalize
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized

# Streamlit interface
st.title("Plant Disease Detection System")

# Step 1: Select the plant type
plant_type = st.selectbox("Select Plant Type", ['Apple', 'Tomatoes'])

# Step 2: Upload image
uploaded_file = st.file_uploader(f"Upload an image of {plant_type} leaf", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=f"Uploaded {plant_type} leaf", use_column_width=True)

    # Step 3: Preprocess image and predict
    preprocessed_image = preprocess_image(image)
    
    # Load the appropriate model for the selected plant type
    model = models[plant_type]
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    disease_name = disease_classes[plant_type][predicted_class]
    confidence_scores = prediction[0] * 100  # Convert to percentages

    # Step 4: Display prediction and confidence score
    st.write(f"Prediction: **{disease_name}**")
    st.write(f"Confidence: **{confidence_scores[predicted_class]:.2f}%**")

    # Step 5: Display additional disease information
    st.subheader("Disease Information")
    st.write(f"**Description**: {disease_info[plant_type][disease_name]['description']}")
    st.write(f"**Precautions**: {disease_info[plant_type][disease_name]['precautions']}")
    st.write(f"**Treatment**: {disease_info[plant_type][disease_name]['treatment']}")

    # Step 6: Visualize confidence scores using a bar chart
    st.subheader("Confidence Score Visualization")
    fig, ax = plt.subplots()
    sns.barplot(x=list(disease_classes[plant_type].values()), y=confidence_scores, ax=ax)
    ax.set_title('Confidence Scores for Predicted Diseases')
    ax.set_ylabel('Confidence (%)')
    st.pyplot(fig)

    # Option to upload another image
    if st.button('Try Another Image'):
        st.experimental_rerun()

# Option to reset the entire interface
if st.button("Reset"):
    st.experimental_rerun()
