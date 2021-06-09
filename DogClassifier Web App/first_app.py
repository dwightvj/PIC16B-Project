import os
import pathlib
import requests
from io import BytesIO

####*IMPORTANT*: Have to do this line *before* importing tensorflow
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

# ensure repeatability in our predictions
def reset_random_seeds():
    '''
    generate random seed to ensure same predictions are outputted upon user input!
    '''
    os.environ['PYTHONHASHSEED'] = str(1)
    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)

# load in our model
model = tf.keras.models.load_model('tf2model_deprecated_newest.h5')

sheet_url = 'https://docs.google.com/spreadsheets/d/135uA2hSgPFbCKOwFMG2xkn_nG_WWSIXdzoSrQawdz9s/edit#gid=2044474371'
url_1 = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
df = pd.read_csv(url_1)

# read in pickle file containing a dicionary of dog breed as keys and an index referring to the dog breed as values
pkl_file = open('all_dog_breeds.pkl', 'rb')
google_form_dict = pickle.load(pkl_file)


def create_y1(breed):
    '''
    :param breed: dog breed
    :return: zero array with a 1 filled at a certain index to indicate breed in question (represents our "class")
    '''

    # we can predict a total of 121 breeds, hence a zero-array of 121 zeros
    zero_array = np.zeros(121)
    num = google_form_dict[breed]
    zero_array[num] = 1

    return zero_array

# online learning
wrong_predictions = df.loc[df.iloc[:, 1] == 'No, all predicted classes were incorrect']
if wrong_predictions.shape[0] > 0:
    # optimizers have built in assumptions that decay lr over time
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    x1_list = []
    y1_list = []
    for index, row in wrong_predictions.iterrows():
        url = row[3].replace('open', 'uc')
        response = requests.get(url)
        # re-fit model to predictions that were done incorrectly
        my_image = Image.open(BytesIO(response.content))
        # downsample user input
        new_image = my_image.resize((224, 224), Image.LANCZOS)
        data = img_to_array(new_image)
        dog = row[2]
        if dog != 'Other Breed':
            x1_list.append([data])
            y1_list.append(create_y1(dog))

    # create our x and y datasets to be refit below
    x1 = np.vstack(x1_list)
    y1 = np.vstack(y1_list)

    # use 3 epochs for the sake of speed
    model.fit(x1, y1, epochs=3, verbose=0)

# read in our class names pickle (acts as our label encoder)
pkl_file = open('class_names.pkl', 'rb')
class_names = pickle.load(pkl_file)

# used as image transformer to make data more friendly
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
    return: top 3 classes (dog breeds) based on user-image
   '''

    data = img_to_array(img)
    samples = expand_dims(data, 0)
    it = train_datagen.flow(samples, batch_size=1)
    pred = model.predict(it)
    # get the top 3 classes in order of most to least likely
    indices = pred[0].argsort()[-3:][::-1]
    return [re.split(r'(\d+)-', class_names[indices[i]])[-1] for i in range(len(indices))]


def main():
    st.header("Predict Dog Breed Using Your Image")

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

        st.header("**Give Us Feedback Below!**")

        st.markdown("""
        <iframe src="https://docs.google.com/forms/d/e/1FAIpQLSfSIVQCYG-7FqWtXbn3iRzFUdOjyigfK1D_HtDrM1bfFO-YXg/viewform?embedded=true" width="700" height="520" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>
        """, unsafe_allow_html=True)