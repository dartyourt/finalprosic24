# DeonScalp: Web Deteksi Penyakit Kulit Kepala
This project aims to develop an IoT and AI-based scalp disease detection system using an ESP32 microcontroller. The system will collect scalp image data using a camera activated by a LIDAR sensor and a buzzer, then analyze the data using machine learning to provide diagnosis and treatment recommendations.

as note, it only can run locally!

## DISCLAIMER 

you have to installed some libraries needed:

+ cv2
+ firebase
+ firebase-admin
+ streamlit
+ google.generativeai
+ tensorflow

etc...

## NOTES
1. firebasedata2.ipynb = running firebase to get and upload data to firebase from sensor
2. scalp_learn.ipynb   = model that had been created for scalp classification
3. deons.py            = streamlit web for running the app locally (cause it needs to access camera hardware
4. Dataset-Image       = dataset that we use for training the model
5. datatrain           = data train for predictions
