import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import firebase_admin
from firebase_admin import credentials, db
from PIL import Image
import google.generativeai as genai
from IPython.display import Markdown, clear_output, display
import time

if not firebase_admin._apps:
    cred = credentials.Certificate("credentials.json") #you must have the credentials
    firebase_admin.initialize_app(cred, {"databaseURL": "https://xx"}) #and the url to the database

# Set up Firebase reference
ref = db.reference('/')

# Google Generative AI configuration
API_KEY = 'Use your API_KEY'
prompt = 'You are an observer and expertise of scalp disease, provide the type of scalp condition, its underlying causes, and some simple, practical tips or tasks to help manage or alleviate the condition from the image based on your expertise.'

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
        pil_image = Image.fromarray(img)
        print("Image processed.")

        genai.configure(api_key=API_KEY)
        print("Analyzing image...")

        model = genai.GenerativeModel(model_name='gemini-1.5-flash')  # or gemini-1.5-pro
        response = model.generate_content([prompt, pil_image], stream=True)
        print("Analysis completed.")

        buffer = []
        for chunk in response:
            for part in chunk.parts:
                buffer.append(part.text)
            clear_output()
            display(Markdown(''.join(buffer)))
            # Display the analysis result in Streamlit
            st.markdown(''.join(buffer))
     

def main():
    st.title("Scalp Condition Capture")

    st.write("""
    ## Instructions:
    1. Ensure you have adequate lighting.
    2. Position your head in front of the camera.
    3. When the distance is under 10, a countdown will start.
    4. The image will be captured automatically after the countdown.
    """)

    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    if webrtc_ctx.video_transformer:
        if webrtc_ctx.video_transformer.start_stream:
            st.write("Adjust your head position in front of the camera.")
        else:
            st.write("Waiting for the distance to be less than 10...")

if __name__ == "__main__":
    main()
