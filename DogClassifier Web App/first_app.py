# import tensorflow as tf
# from numpy import expand_dims
# from PIL import Image
# from tensorflow import keras
# from tensorflow.keras import applications
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, Activation
# import streamlit as st
# import pickle
# import re
# import timeit

#######################################################
import os
import pathlib
import skimage

####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED'] = str(1)
import tensorflow as tf
from numpy import expand_dims
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import streamlit as st
import pickle
import re
import timeit
import random
import numpy as np
import time
from skimage import measure


def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(1)
    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)


#######################################################

model = tf.keras.models.load_model('my_model.h5')

pkl_file = open('class_names.pkl', 'rb')
class_names = pickle.load(pkl_file)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2,
)


def make_prediction(img):
    '''
    This function takes in a image and model, and uses the model to predict the class of the image
   '''
    data = img_to_array(img)
    samples = expand_dims(data, 0)
    it = train_datagen.flow(samples, batch_size=1)
    pred = model.predict(it)
    indices = pred[0].argsort()[-3:][::-1]
    return [re.split(r'(\d+)-', class_names[indices[i]])[-1] for i in range(len(indices))]


def main():
    st.header("Predict Dog Breed")

    reset_random_seeds()
    image = st.file_uploader("The image of your dog!", ["png", "jpg", "jpeg"], key='file')
    starttime = timeit.default_timer()

    if image is not None:  # if the image is an actual file then
        col1, col2 = st.beta_columns(2)  # split our layout into two columns
        with col1:  # the first column in our layout will display the image
            image_to_share = Image.open(image)
            resized_image = image_to_share.resize((224, 224), Image.ANTIALIAS)
            # tf.keras.backend.clear_session()
            st.image(resized_image, width=265)
        with col2:
            st.write("## Top Predicted Classes Are:")
            if pathlib.Path(str(image.name)).suffix.lower() == '.png':
                resized_image = resized_image.convert('RGB')

            with st.spinner('Loading...'):
                predicted_class = make_prediction(resized_image)
                l1 = [breed.replace('_', ' ').replace('-', ' ') for breed in predicted_class]
                l2 = [re.sub(r'\b[a-z]', lambda m: m.group().upper(), i) for i in l1]
                l3 = [breed.replace('And', 'and') for breed in l2]
                st.write("Load/Compile Time (in seconds) :", timeit.default_timer() - starttime)
                st.write('## **1.{}**'.format(l3[0]))
                st.write('## **2.{}**'.format(l3[1]))
                st.write('## **3.{}**'.format(l3[2]))
