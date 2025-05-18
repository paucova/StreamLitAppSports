import streamlit as st
from PIL import Image
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# we load the model using the load_model library from keras
model = load_model("sport_model.keras")
# we get the class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# in our app, create something simple
st.title("Sport Classification")
# option to upload a file from computer
uploaded_image = st.file_uploader("Please upload an image of a sport", type=["jpg", "jpeg", "png"])

# if the user uploaded an image, convert to the format RGB
if uploaded_image is not None:
    img = Image.open(uploaded_image).convert("RGB")
    
    # in here we resize so that the model can look at it 
    img = img.resize((224, 224))
    
    # these are more processing things for the image to be able to be read by the model
    image_array = image.img_to_array(img)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array) #scaling for mobilenet
    
    # get the prediction from the model
    predictions = model.predict(image_array)
    
    #get the first one
    predicted_sport = np.argmax(predictions[0])
    score = np.max(predictions[0])*100
    
    # print it with the score
    st.write(f"Predicted sport is: {class_names[predicted_sport]}: with ({score:.2f}% confidence)")
    if score > 95:
        st.balloons()