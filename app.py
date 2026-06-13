import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("plant_disease_classifier.keras")

# Class names
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

st.set_page_config(
    page_title="Plant Disease Prediction",
    page_icon="🌿"
)

st.title("🌿 Plant Disease Prediction using CNN")

st.write(
    "Upload a plant leaf image and the model will predict the disease category."
)

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    # Preprocessing
    image = image.resize((224, 224))
    image_array = np.array(image)

    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)

    disease = class_names[predicted_index]

    st.success(f"Prediction: {disease}")
    st.write(f"Confidence Score: {confidence:.2%}")

    st.subheader("Prediction Probabilities")

    probability_dict = {
        class_names[i]: float(prediction[0][i])
        for i in range(len(class_names))
    }

    st.bar_chart(probability_dict)