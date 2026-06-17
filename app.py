import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your model (Keras 2 standard)
@st.cache_resource
def load_my_model():
    # Keras 2 handles .h5 files natively
    model = tf.keras.models.load_model('waste_speration.h5')
    return model

model = load_my_model()

# Ensure these match the EXACT alphabetical order of your training folders
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

st.title("Waste Classification App ♻️")
st.write("Upload an image of waste to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# --- Developed By Section ---
st.markdown("---")
st.write("### Developed By")
st.write("**Gaurav Gupta** | BIT Mesra, AIML")

# LinkedIn link with icon
st.markdown(
    '<a href="https://www.linkedin.com/in/gaurav-gupta-79754a377/" target="_blank">'
    '<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" style="vertical-align: middle;"/>'
    ' My LinkedIn Profile</a>', 
    unsafe_allow_html=True
)
st.markdown("---")
# ----------------------------

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocessing
    # Ensure this matches the resize used during your training
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Using numpy for dimension expansion
    
    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    st.write(f"### Predicted Category: {predicted_class}")
    st.write(f"### Confidence: {confidence:.2f}%")
