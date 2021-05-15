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
####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(1)
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

def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(1)
   tf.random.set_seed(1)
   np.random.seed(1)
   random.seed(1)
#######################################################

model = tf.keras.models.load_model('my_model.h5')
loss = 'categorical_crossentropy'
optimizer = tf.keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

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

def main():
    reset_random_seeds()
    # Allow the user to upload a image of their dog
    # tf.keras.backend.clear_session()
    image = None
    image = st.file_uploader("The image of your dog!", ["png", "jpg", "jpeg", 'HEIC'], key='file')

    starttime = timeit.default_timer()

    # tf.keras.backend.clear_session()
    # img_width, img_height = 224, 224
    # channels = 3
    # InceptionV3 = applications.InceptionV3(include_top=False, input_shape=(img_width, img_height, channels),
    #                                        weights='imagenet')
    # def build_model():
    #     model = Sequential()
    #     model.add(InceptionV3)
    #     model.add(GlobalAveragePooling2D())
    #     model.add(Dropout(0.2))
    #     model.add(Dense(120, activation='softmax'))
    #     model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    #     return model

    def make_prediction(img):
        '''
        This function takes in a image and model, and uses the model to predict the class of the image 
       '''
        # img = img.convert('RGB')
        # img = img.resize((224, 224), Image.NEAREST)
        # img = tf.keras.preprocessing.image.img_to_array(img)
        # img_array = tf.expand_dims(img, 0)
        # prediction = model.predict(img_array)
        # pred_class = np.argmax(prediction)
        # return re.split('(\d+)-', classes[pred_class])[-1]

        # tf.keras.backend.clear_session()
        # model = build_model()
        # model.load_weights('dog_modelv2.weights')

        # st.write("Load/Compile Time (in seconds) :", timeit.default_timer() - starttime)
        # st.write(image.name)
        # img = img.convert('RGB')
        #foo = img.resize((224, 224), Image.ANTIALIAS)
        # foo.save(str(image.name))
        # st.write(os.stat(str(image.name)).st_size)
        data = img_to_array(img)
        # new_data = tf.image.resize(data, [128, 128])
        # new_data = tf.image.resize(data, [224, 224])
        samples = expand_dims(data, 0)
        it = train_datagen.flow(samples, batch_size=1)
        pred = model.predict(it)
        score = tf.nn.softmax(pred[0])
        score_array = score.numpy()
        indices = score_array.argsort()[-3:][::-1]
        # tf.keras.backend.clear_session()
        st.write("Load/Compile Time (in seconds) :", timeit.default_timer() - starttime)
        # return re.split(r'(\d+)-', class_names[indices[0]])[-1]
        return [re.split(r'(\d+)-', class_names[indices[i]])[-1] for i in range(len(indices))]

    if image is not None:  # if the image is an actual file then
        col1, col2 = st.beta_columns(2)  # split our layout into two columns
        # st.balloons()  # display the ballons
        with col1:  # the first column in our layout will display the image
            image_to_share = Image.open(image)
            resized_image = image_to_share.resize((224, 224), Image.ANTIALIAS)
            # tf.keras.backend.clear_session()
            st.image(resized_image, width=265)
        with col2:
            #st.write("## The Predicted Class Is:")
            st.write("## Top Predicted Classes Are:")
            predicted_class = make_prediction(resized_image)
            st.write('# 1.{}'.format(predicted_class[0]))
            st.write('# 2.{}'.format(predicted_class[1]))
            st.write('# 3.{}'.format(predicted_class[2]))
            image = None