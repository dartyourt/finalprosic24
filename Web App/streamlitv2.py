import streamlit as st
import cv2
from PIL import Image
import google.generativeai as genai
import firebase_admin
from firebase_admin import db, credentials
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Authenticate to Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("credentials.json")
    firebase_admin.initialize_app(cred, {"databaseURL": "https://scalp-detection-default-rtdb.asia-southeast1.firebasedatabase.app/"})

# Set up the reference to the database path you want to listen to
ref = db.reference('/')

# Load the scalp condition classifier model
model = load_model('scalp_condition_classifier_model.h5', compile=False)
img_height, img_width = 150, 150  # Image dimensions should match the model input

# Attach the listener to the reference
def on_snapshot(event):
    print(event.event_type)  # can be 'put' or 'patch'
    print(event.path)  # relative to the reference, it can be '' for the whole path or a sub-path
    print(event.data)  # new data at that location
ref.listen(on_snapshot)

# Function to take a picture using USB camera
def take_picture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open video device.")
        return None
    ret, frame = cap.read()
    if ret:
        cap.release()
        return frame
    cap.release()
    return None

# Streamlit app
st.title("Scalp Condition Detection")
st.write("This web app allows you to detect scalp conditions based on images.")

st.header("Instructions")
st.write("""
1. Ensure your webcam is connected.
2. Press the 'Start' button to capture an image.
3. The captured image will be analyzed for scalp conditions.
4. Wait for the results to be displayed below.
""")

if st.button("Start"):
    st.write("Capturing image...")
    image = take_picture()
    if image is not None:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")
        
        # Save the image to disk
        image_path = 'captured_image.jpg'
        cv2.imwrite(image_path, image)

        # Load the image and process it
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((img_height, img_width))
        img_array = tf.keras.preprocessing.image.img_to_array(pil_image)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        print("Analyzing image with scalp condition classifier model...")
        predictions = model.predict(img_array)
        condition = np.argmax(predictions, axis=1)

        condition_mapping = {0: 'Alopecia Areata', 1: 'Seborrhoeic Dermatitis', 2: 'Scalp Psoriasis', 3: 'Tinea Capitis', 4: 'Normal'}
        condition_name = condition_mapping.get(condition[0], "Unknown")

        print(f"Condition: {condition_name}")
        st.markdown(f"Condition: {condition_name}")
        
        # Google Generative AI configuration
        API_KEY = 'AIzaSyCxwAg3w7e92R10w8eEQg55AahWBqxEfKM'
        prompt = 'You are an observer and expertise of scalp disease, give the suggestion for the condition and some simple, practical tips or tasks to help manage or alleviate the condition'
        
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        response = model.generate_content([prompt, condition_name], stream=True)

        buffer = []
        for chunk in response:
            for part in chunk.parts:
                buffer.append(part.text)
        
        st.markdown(''.join(buffer))
    else:
        st.error("Failed to capture image.")

# Keep the main thread alive to keep listening to updates
try:
    while True:
        pass
except KeyboardInterrupt:
    st.write("Stopped listening for database changes.")
