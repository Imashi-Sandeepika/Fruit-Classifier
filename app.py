import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

st.set_page_config(page_title="Fruit classifier", page_icon="🍎")

# Load the saved model
@st.cache_resource
def load_model_v3():
    return keras.models.load_model('fruit_classifier_model.keras')
 
try:
    model = load_model_v3()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Class labels as defined in the training script
class_labels = ['Apples', 'Mangoes']

st.title("🍎🥭 Fruit Classifier")
st.write("Upload an image of an Apple or a Mango, and the model will predict which one it is!")

uploaded_file = st.file_uploader("Drag and drop an image here...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)
    
    if model_loaded:
        with st.spinner('Classifying...'):
            # Preprocess the image exactly as done in training
            img = image.resize((150, 150))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            
            # Predict
            prediction = model.predict(img_array)
            predicted_idx = np.argmax(prediction[0])
            raw_score = prediction[0][predicted_idx]
            confidence = raw_score * 100
            
            # The class labels from the generator are ordered alphabetically: ['Apples', 'Mangoes', 'Other']
            class_labels = ['Apples', 'Mangoes', 'Other']
            predicted_class = class_labels[predicted_idx]
            
            if predicted_class == 'Other':
                st.warning(f"**Prediction:** Not match (Does not look like an Apple or Mango)")
            else:
                st.success(f"**Prediction:** {predicted_class}")
                
            st.info(f"**Confidence:** {confidence:.2f}%")
