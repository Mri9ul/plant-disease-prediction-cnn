# 🌿 Plant Disease Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

An end-to-end computer vision project that classifies 38 different plant diseases using leaf images. Built with a custom Convolutional Neural Network (CNN) and deployed as an interactive web application using Streamlit.

---

## 📌 Project Overview
Agricultural diseases cause significant losses in crop yield worldwide. This project leverages deep learning to automate the diagnosis of plant diseases. By simply uploading a photo of a leaf, the trained model can instantly predict whether the plant is healthy or suffering from a specific disease, along with a confidence score.

### Key Features
* **Custom CNN Architecture:** Built from scratch using TensorFlow/Keras, featuring Batch Normalization, MaxPooling, and Dropout layers to prevent overfitting.
* **Extensive Dataset:** Trained on the **PlantVillage Dataset**, encompassing 38 distinct classes of plant leaves (both healthy and diseased).
* **Interactive UI:** A lightweight, user-friendly Streamlit web app for real-time predictions.
* **Evaluation Metrics:** Includes accuracy/loss tracking, classification reports, and confusion matrix visualizations.

---

## 📂 Project Structure
```text
plant-disease-prediction-cnn/
│
├── app.py                             # Streamlit web application script
├── class_indices.json                 # JSON mapping of class indices to disease names
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── notebooks/
    └── plant_disease_prediction_cnn.ipynb  # Exploratory Data Analysis & Model Training
