import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image
import numpy as np
import os, time, zipfile
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array


st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        color: #4A90E2;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #555;
        margin-top: 5px;
        margin-bottom: 30px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #888;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>MRI Tumor Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload your MRI scan image and get instant results</div>", unsafe_allow_html=True)



upload_file = st.file_uploader('Enter the MRI Scan Image',
                                type = ['jpg','jpeg','png'],
                                label_visibility='hidden')


st.write("Current directory:", os.getcwd())
st.write("Files:", os.listdir())

try:
    with zipfile.ZipFile("mrimodel.keras") as zf:
        st.write("✅ Model is a valid .keras zip file")
except zipfile.BadZipFile:
    st.write("❌ Not a valid .keras zip file (still pointer?)")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mrimodel.keras')
model = load_model()



classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

meningioma = 'meningioma.jpg'
glioma = 'glioma.jpg'
Notumour = 'Notumour.jpg'

def pre(img):
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis = 0)
    return img_arr

if upload_file is not None:
    img = Image.open(upload_file)
    st.image(img, caption='Uploaded image',width=400)

    if st.button('Predict'):
        img = pre(img)
        pred = model.predict(img)
        with st.spinner("Analyzing the Results", show_time=True):
             time.sleep(5)
        pred_class = classes[np.argmax(pred, axis=1)[0]]
        confidence = np.max(pred)*100



        st.markdown("""
            <style>
            .result-box {
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                margin-top: 20px;
                text-align: center;
            }
            .result-title {
                color: #4A90E2;
                font-size: 28px;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .result-text {
                font-size: 20px;
                color: #333333;
            }
            .confidence {
                font-size: 18px;
                color: #888888;
                margin-top: 10px;
            }
            </style>
        """, unsafe_allow_html=True)



        st.markdown(f"""
            <div class="result-box">
                <div class="result-title">Prediction</div>
                <div class="result-text">Predicted class : <b>{pred_class}</b></div>
                <div class="confidence">Confidence : {confidence:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)







