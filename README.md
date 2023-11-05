# Real-time-Hand-Gesture-Recognition-and-Translation

Libraries Needed
Install the following by using ```pip install```
1. tensorflow
2. mediapipe
3. opencv-python
4. numpy
5. googletrans==4.0.0-rc1
   
**Dataset and training:**
1. Pre-trained model is used here 
2. Mediapipe dataset is used.

**Mediapipe**

MediaPipe is an open-source framework for building pipelines to perform computer vision inference over arbitrary sensory data such as video or audio. Using MediaPipe, such a perception pipeline can be built as a graph of modular components.

Run the hand gesture.ipynb notenook to get a prediction alone 

To run the web app with translation to other language

```streamlit run app.py```

The app translates the predictions into 4 languages 
1. German
2. French
3. Tamil
4. Spanish
