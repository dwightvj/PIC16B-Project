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
import requests
from io import BytesIO

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
import pandas as pd



def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(1)
    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)

dog_dict = {
'Chihuahua1.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/chihuahua/Chihuahua1.jpg',
'Beagle1.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/beagle/Beagle1.jpg',
'Greyhound0.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/greyhound/Greyhound0.jpg',
'Old English Sheepdog1.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/old-english-sheepdog/Old%20English%20Sheepdog1.jpg',
'Bloodhound0.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/bloodhound/Bloodhound0.jpg',
'Bullmastiff0.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/bullmastiff/Bullmastiff0.jpg',
'Golden Retriever0.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/golden-retriever/Golden%20Retriever0.jpg',
'Chihuahua2.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/chihuahua/Chihuahua2.jpg',
'Bloodhound1.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/bloodhound/Bloodhound1.jpg',
'Pomeranian0.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/pomeranian/Pomeranian0.jpg',
'Brittany0.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/brittany/Brittany0.jpg',
'Kuvasz0.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/kuvasz/Kuvasz0.jpg',
'Old English Sheepdog0.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/old-english-sheepdog/Old%20English%20Sheepdog0.jpg',
'Golden Retriever1.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/golden-retriever/Golden%20Retriever1.jpg',
'Pomeranian2.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/pomeranian/Pomeranian2.jpg',
'Shih Tzu0.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/shih-tzu/Shih%20Tzu0.jpg',
'German Shepherd Dog0.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/german-shepherd-dog/German%20Shepherd%20Dog0.jpg',
'Siberian Husky0.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/siberian-husky/Siberian%20Husky0.jpg',
'Great Dane1.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/great-dane/Great%20Dane1.jpg',
'Tibetan Mastiff0.jpg' : 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/dog_photos/tibetan-mastiff/Tibetan%20Mastiff0.jpg'
}

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

    # img = Image.open(generate(img))
    data = img_to_array(img)
    samples = expand_dims(data, 0)
    it = train_datagen.flow(samples, batch_size=1)
    pred = model.predict(it)
    indices = pred[0].argsort()[-3:][::-1]
    return [re.split(r'(\d+)-', class_names[indices[i]])[-1] for i in range(len(indices))]


def main():
    st.header("Predict Dog Breed Using a Sample Image")
    reset_random_seeds()
    option = st.selectbox('Select an Image!', ('Shih Tzu0.jpg', 'Chihuahua1.jpg', 'Golden Retriever1.jpg',
                                               'Greyhound0.jpg', 'Old English Sheepdog1.jpg', 'Bloodhound0.jpg',
                                               'Bullmastiff0.jpg', 'Siberian Husky0.jpg', 'Old English Sheepdog0.jpg',
                                               'Bloodhound1.jpg', 'Pomeranian0.jpg', 'Brittany0.jpg', 'Kuvasz0.jpg',
                                               'Chihuahua2.jpg', 'Golden Retriever1.jpg', 'Pomeranian2.jpg',
                                               'Golden Retriever0.jpg', 'Great Dane1.jpg', 'Tibetan Mastiff0.jpg',
                                               'Beagle1.jpg'))
    if st.button('Submit', key='2'):
        url = dog_dict[option]
        image = Image.open(requests.get(url, stream=True).raw)
        starttime = timeit.default_timer()

        if image is not None:  # if the image is an actual file then
            col1, col2 = st.beta_columns(2)  # split our layout into two columns
            with col1:  # the first column in our layout will display the image
                image_to_share = image
                resized_image = image_to_share.resize((224, 224), Image.ANTIALIAS)
                # tf.keras.backend.clear_session()
                st.image(resized_image, width=265)
            with col2:
                st.write("## Top Predicted Classes Are:")
                # if pathlib.Path(str(image.name)).suffix.lower() == '.png':
                #     resized_image = resized_image.convert('RGB')

                with st.spinner('Loading...'):
                    predicted_class = make_prediction(resized_image)
                    l1 = [breed.replace('_', ' ').replace('-', ' ') for breed in predicted_class]
                    l2 = [re.sub(r'\b[a-z]', lambda m: m.group().upper(), i) for i in l1]
                    l3 = [breed.replace('And', 'and') for breed in l2]
                    st.write("Load/Compile Time (in seconds) :", timeit.default_timer() - starttime)
                    st.write('## **1.{}**'.format(l3[0]))
                    st.write('## **2.{}**'.format(l3[1]))
                    st.write('## **3.{}**'.format(l3[2]))




