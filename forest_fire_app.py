import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array
import requests
from PIL import Image
from io import BytesIO

# Define paths to your image dataset directories
fire_dir = "DS/fire/"
nofire_dir = "DS/nofire/"

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size=(150, 150)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = (img * 255).astype(np.uint8)  # Convert back to uint8
    return img

# Load image paths
fire_image_paths = [os.path.join(fire_dir, filename) for filename in os.listdir(fire_dir)]
nofire_image_paths = [os.path.join(nofire_dir, filename) for filename in os.listdir(nofire_dir)]

# Calculate class distribution
fire_count = len(fire_image_paths)
nofire_count = len(nofire_image_paths)

# Calculate average image dimensions
image_dimensions = []
for image_path in fire_image_paths + nofire_image_paths:
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    image_dimensions.append((height, width))

avg_height = np.mean([dim[0] for dim in image_dimensions])
avg_width = np.mean([dim[1] for dim in image_dimensions])

# Calculate average color values
avg_colors_fire = defaultdict(list)
avg_colors_nofire = defaultdict(list)

for image_path in fire_image_paths:
    img = load_and_preprocess_image(image_path)
    avg_color = np.mean(img, axis=(0, 1))
    for i in range(3):
        avg_colors_fire[i].append(avg_color[i])

for image_path in nofire_image_paths:
    img = load_and_preprocess_image(image_path)
    avg_color = np.mean(img, axis=(0, 1))
    for i in range(3):
        avg_colors_nofire[i].append(avg_color[i])

# Calculate aspect ratios
aspect_ratios_fire = [dim[0] / dim[1] for dim in image_dimensions[:fire_count]]
aspect_ratios_nofire = [dim[0] / dim[1] for dim in image_dimensions[fire_count:]]

# Investigate grayscale images
grayscale_count = 0
for image_path in fire_image_paths + nofire_image_paths:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if len(img.shape) < 3:
        grayscale_count += 1

# Define function to display EDA graphs in Streamlit
def display_eda_graphs():
    # Plot aspect ratios
    st.subheader("Aspect Ratios")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(aspect_ratios_fire, color='red', alpha=0.5, label="Fire")
    plt.hist(aspect_ratios_nofire, color='blue', alpha=0.5, label="No Fire")
    plt.xlabel("Aspect Ratio")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Aspect Ratios")
    plt.subplot(1, 2, 2)
    plt.bar(["Grayscale", "Color"], [fire_count + nofire_count - grayscale_count, grayscale_count, ])
    plt.ylabel("Number of Images")
    plt.title("Grayscale vs. Color Images")
    st.pyplot(plt)

    # Display image histograms
    st.subheader("Image Histograms")
    sample_image_path = fire_image_paths[0]  # Choose a sample image for histogram analysis
    sample_image = load_and_preprocess_image(sample_image_path)
    sample_image_gray = cv2.cvtColor(sample_image, cv2.COLOR_RGB2GRAY)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(sample_image)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.hist(sample_image.ravel(), bins=256, color='blue', alpha=0.7)
    plt.title("RGB Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.subplot(1, 3, 3)
    plt.hist(sample_image_gray.ravel(), bins=256, color='gray', alpha=0.7)
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    st.pyplot(plt)

    # Average color values
    st.subheader("Average Color Values")
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.hist(avg_colors_fire[i], color='red', alpha=0.5, label="Fire")
        plt.hist(avg_colors_nofire[i], color='blue', alpha=0.5, label="No Fire")
        plt.xlabel(f"Average {['Red', 'Green', 'Blue'][i]} Channel Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title(f"Average {['Red', 'Green', 'Blue'][i]} Channel Value Distribution")
    st.pyplot(plt)

    # Color channel correlations
    st.subheader("Color Channel Correlations")
    def plot_color_correlations(image_paths, title):
        color_correlations = []

        for image_path in image_paths:
            img = load_and_preprocess_image(image_path)
            r, g, b = cv2.split(img)
            corr_rg = np.corrcoef(r.ravel(), g.ravel())[0, 1]
            corr_rb = np.corrcoef(r.ravel(), b.ravel())[0, 1]
            corr_gb = np.corrcoef(g.ravel(), b.ravel())[0, 1]
            color_correlations.append([corr_rg, corr_rb, corr_gb])

        color_correlations = np.array(color_correlations)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.scatter(color_correlations[:, 0], color_correlations[:, 1], c='red', label="Fire")
        plt.scatter(color_correlations[:, 0], color_correlations[:, 1], c='blue', label="No Fire")
        plt.xlabel("Correlation (R-G)")
        plt.ylabel("Correlation (R-B)")
        plt.legend()
        plt.title(title)

        plt.subplot(1, 3, 2)
        plt.scatter(color_correlations[:, 0], color_correlations[:, 2], c='red', label="Fire")
        plt.scatter(color_correlations[:, 0], color_correlations[:, 2], c='blue', label="No Fire")
        plt.xlabel("Correlation (R-G)")
        plt.ylabel("Correlation (G-B)")
        plt.legend()
        plt.title(title)

        plt.subplot(1, 3, 3)
        plt.scatter(color_correlations[:, 1], color_correlations[:, 2], c='red', label="Fire")
        plt.scatter(color_correlations[:, 1], color_correlations[:, 2], c='blue', label="No Fire")
        plt.xlabel("Correlation (R-B)")
        plt.ylabel("Correlation (G-B)")
        plt.legend()
        plt.title(title)

        st.pyplot(plt)

    plot_color_correlations(fire_image_paths, "Color Channel Correlations (Fire)")
    plot_color_correlations(nofire_image_paths, "Color Channel Correlations (No Fire)")

    # Sample images
    st.subheader("Sample Images")
    plt.figure(figsize=(12, 4))
    for i, image_path in enumerate(fire_image_paths[:3] + nofire_image_paths[:3]):
        img = load_and_preprocess_image(image_path)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title("Fire" if i < 3 else "No Fire")
        plt.axis("off")
    st.pyplot(plt)

# Streamlit UI
st.title("Forest Fire Prediction")

# Display EDA graphs
st.write("Exploratory Data Analysis (EDA)")
display_eda_graphs()

# Load the models
model_paths = {
    "EfficientNetB0": "saved_model/efnet",
    "CNN": "saved_model/CNN",
    "VGG16": "saved_model/vgg16_model_top"
    # Update with the correct paths
    # Add more models here if needed
}

models = {name: tf.keras.models.load_model(path) for name, path in model_paths.items()}

# Accuracy graph image paths
accuracy_images = {
    "EfficientNetB0": "saved_model/efnet/accuracy.png",
    "CNN": "saved_model/CNN/accuracy.png",
    "VGG16": "saved_model/vgg16_model_top/accuracy.png"
}

# Model selection
model_name = st.selectbox("Select a model", list(models.keys()))

# Selected model
model = models[model_name]

# Function to predict and display images
def predict_image(image):
    if isinstance(image, str):  # Local file path
        img = keras.preprocessing.image.load_img(image, target_size=(250, 250))
    else:  # Uploaded file content
        img = Image.open(image)
        img = img.resize((250, 250))

    plt.imshow(img)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    if prediction[0] >= 0.5:
        result = "Fire"
    else:
        result = "No Fire"
    plt.xlabel(result, fontsize=30)
    st.write(f"Prediction: {result}")

# Upload image for prediction or paste URL
uploaded_file = st.file_uploader("Choose an image or paste URL...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    if st.button("Start New Training Session and Delete Model"):
        # Code to start a new training session and delete the model
        st.write("New training session started. Model deleted.")
    else:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Perform prediction
        predict_image(uploaded_file)

# Display accuracy graph for the selected model
st.write("Accuracy Graph")
accuracy_image_path = accuracy_images.get(model_name)
if accuracy_image_path is not None:
    st.image(accuracy_image_path, use_column_width=True)
else:
    st.write("Accuracy graph not available for the selected model.")
