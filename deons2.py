import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import firebase_admin
from firebase_admin import credentials, db
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

if not firebase_admin._apps:
    cred = credentials.Certificate("credentials.json")
    firebase_admin.initialize_app(cred, {"databaseURL": "https://scalp-detection-default-rtdb.asia-southeast1.firebasedatabase.app/"})

# Set up Firebase reference
ref = db.reference('/')

# Initialize Streamlit session state
if "condition_name" not in st.session_state:
    st.session_state.condition_name = None

# Load the scalp condition classifier model
model = load_model('scalp_condition_classifier_model.h5', compile=False)
img_height, img_width = 150, 150  # Image dimensions should match the model input

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.capture = False
        self.start_stream = False
        self.countdown_started = False
        self.countdown_time = 5  # Countdown time in seconds

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Dummy distance calculation for illustration purposes
        distance = self.calculate_distance(img)
        cv2.putText(img, f"Distance: {distance}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if distance < 10:
            self.start_stream = True
            if not self.countdown_started:
                self.countdown_start_time = time.time()
                self.countdown_started = True
        else:
            self.start_stream = False
            self.countdown_started = False

        if self.start_stream and self.countdown_started:
            elapsed_time = time.time() - self.countdown_start_time
            remaining_time = self.countdown_time - elapsed_time

            if remaining_time > 0:
                cv2.putText(img, f"Capturing in {int(remaining_time)} seconds", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                self.capture = True
                self.start_stream = False
                self.countdown_started = False

        if self.capture:
            self.capture = False
            cv2.imwrite("scalp_image.jpg", img)
            self.process_image(img)

        return img

    def calculate_distance(self, img):
        # Dummy distance calculation logic
        # Replace this with actual distance calculation logic
        return 9

    def process_image(self, img):
        print("Image captured and processing...")
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil_image = pil_image.resize((img_height, img_width))
        img_array = tf.keras.preprocessing.image.img_to_array(pil_image)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        print("Analyzing image with scalp condition classifier model...")
        predictions = model.predict(img_array)
        condition = np.argmax(predictions, axis=1)

        condition_mapping = {0: 'Alopecia Areata', 1: 'Seborrhoeic Dermatitis', 2: 'Scalp Psoriasis', 3: 'Tinea Capitis', 4: 'Normal'}
        st.session_state.condition_name = condition_mapping.get(condition[0], "Unknown")

        print(f"Condition: {st.session_state.condition_name}")


def main():
    st.title("Scalp Condition Capture")

    st.write("""
    ## Instructions:
    1. Ensure you have adequate lighting.
    2. Position your head in front of the camera.
    3. When the distance is under 10, a countdown will start.
    4. The image will be captured automatically after the countdown.
    """)

    webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    if webrtc_ctx.video_processor:
        if webrtc_ctx.video_processor.start_stream:
            st.write("Adjust your head position in front of the camera.")
        else:
            st.write("Waiting for the distance to be less than 10...")

    # Display the analysis result in Streamlit
    if st.session_state.condition_name is not None:
        st.markdown(f"**Condition:** {st.session_state.condition_name}")

if __name__ == "__main__":
    main()