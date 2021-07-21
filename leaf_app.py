import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
import streamlit as st
import cv2
class_names=['Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___healthy','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_mosaic_virus','Tomato___Tomato_Yellow_Leaf_Curl_Virus']

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('Tomato_Leaf_Disease_model_inspctionV3_WebApp_Streamlit.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Tomato Leaf Disease Detaction Neural Network App </h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)


file = st.file_uploader("Please upload a Tomato leaf picture", type=["jpg"])
import cv2
from PIL import Image, ImageOps
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
        size = (180,180)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
               
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    if st.button("Predict"):
      predictions = import_and_predict(image, model)
      score=np.array(predictions[0])
      st.title(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
            )
