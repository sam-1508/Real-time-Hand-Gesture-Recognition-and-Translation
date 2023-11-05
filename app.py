import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from googletrans import Translator

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('model')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()

# Define author details
author_name = "Samritha S"
author_email = "samrithasing03@gnail.com"
linkedin_link = "https://www.linkedin.com/in/samritha-singaravelan-7b05ab225"
github_link = "https://github.com/sam-1508"

# Custom CSS for the author details box
css = """
<style>
.author-box {
    background-color: #363636;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0px 0px 5px #888;
}
.author-links a {
    margin-right: 10px;
    text-decoration: none;
}
</style>
"""

# Sidebar with author details in a box
st.sidebar.title("My Details")
st.sidebar.markdown(css, unsafe_allow_html=True)
st.sidebar.markdown(f"<div class='author-box'><h3>{author_name}</h3><div class='author-links'><a href='{linkedin_link}' target='_blank'>LinkedIn</a><br><a href='mailto:{author_email}' target='_blank'>Email</a><br><a href='{github_link}' target='_blank'>GitHub</a></div></div>", unsafe_allow_html=True)

# Initialize the translator
translator = Translator()

def translate_gesture(class_name, target_language='en'):
    try:
        translated = translator.translate(class_name, src='en', dest=target_language)
        return translated.text
    except Exception as e:
        return class_name

def main():
    st.title("Hand Gesture Recognition")
    language = st.selectbox("Select Target Language", ["German", "Tamil", "French", "Spanish", "English"])
    cap = cv2.VideoCapture(0)

    st.frame=st.empty()
    

    translated_gesture = st.empty()

    while True:
        # Read each frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)

        className = ''
        translated_text = ""

        # post-process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Predict gesture
                prediction = model.predict([landmarks])
                classID = np.argmax(prediction)
                className = classNames[classID]

                if language == "German":
                    target_language = 'de'
                elif language == "Tamil":
                    target_language = 'ta'
                elif language == "French":
                    target_language = 'fr'
                elif language == "Spanish":
                    target_language = 'es'
                elif language == "English":
                    target_language = 'en'
                else:
                    target_language = 'en' 

                translated_text = translate_gesture(className, target_language)
                translated_gesture.write(translated_text)
        
        # Show the prediction on the frame
        cv2.putText(frame, f"English: {className}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        if language == "Tamil":
            translated_gesture.text(f"Translated Gesture ({language}): {translated_text}")
        else:
            cv2.putText(frame, f"{language}: {translated_text}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        # Display the frame in Streamlit
        st.frame.image(frame, channels="BGR", use_column_width=True)

if __name__ == "__main__":
    main()
