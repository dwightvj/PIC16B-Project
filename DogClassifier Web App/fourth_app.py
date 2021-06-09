import streamlit as st
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import re
import random
from scipy import spatial

df = pd.read_csv("https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/data/breed_characteristics.csv")

# drop mixed breed
df.drop(89, axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

# only choose top 6 columns of greatest variance, drop the rest
df = df.drop(["name", "dogFriendly", "kidFriendly", "highEnergy", "intelligence", "toleratesHot", "toleratesCold"],
             axis=1)

df['lowShedding'] = df['lowShedding'].replace({1: 5, 2: 4, 3: 3, 4: 2, 5: 1})
df['lowBarking'] = df['lowBarking'].replace({1: 5,2: 4,3: 3,4: 2,5: 1})
df['easyToGroom'] = df['easyToGroom'].replace({1: 5, 2: 4,3: 3,4: 2,5: 1})
df['easyToTrain'] = df['easyToTrain'].replace({1: 5,2: 4,3: 3,4: 2,5: 1})

df.rename(columns={'lowShedding': 'Shedding', 'lowBarking': 'Barking',
                   'easyToGroom': 'Grooming', 'easyToTrain': 'Training'}, inplace=True)

# create new vector column for dogs
df['list'] = df[['size', 'Shedding', 'Grooming',
                 'goodHealth', 'Barking', 'Training']].values.tolist()

# create list of the behavior attributes lists
breeds = df['list'].tolist()
# create KDTree based on these breeds
tree = spatial.KDTree(breeds)

# recommend the top three breeds that are the "nearest neighbor" to input
def top3rec(l):
    '''
    :param l: the vector provided by the user (based on their preferences)
    :return: dog breed
    '''
    # find the indices of the 3 closest vectors to l
    closest_indices = tree.query(l, k=3)[1]

    # get the vectors of attributes of these 3 indices
    dogs_behav = [breeds[i] for i in closest_indices]

    # find the indices containing these attribute vectors
    indices = [breeds.index(dog) for dog in dogs_behav]
    # get the breed names based on index
    name = [df.iloc[index, 0] for index in indices]
    return name


def main():
    st.header("Find Your Perfect Dog")
    # st.write("\n")
    st.write("### *What Do You Look For in a Dog?*")
    size = st.select_slider('1. Size', options=['Petite', 2, 3, 4, 'Large'], value=3)
    shedding = st.select_slider('2. Shedding', options=['Very Little', 2, 3, 4, 'Excessive'], value=2)
    grooming = st.select_slider('3. Amount of Maintenance (Grooming)', options=['Low', 2, 3, 4, 'High'], value=3)
    health = st.select_slider('4. Health', options=['Poor', 2, 3, 4, 'Good'], value=4)
    barking = st.select_slider('5. Barking', options=['Quiet', 2, 3, 4, 'Loud'], value=2)
    training = st.select_slider('6. Ability to Train', options=['Easy', 2, 3, 4, 'Difficult'], value=3)

    if st.button('Submit', key='1'):
        st.write("\n")
        st.write("\n")
        user_input = [size, shedding, grooming, health, barking, training]

        dct = {'Petite': 1, 'Large': 5, 'Very Little': 1, 'Excessive': 5, 'Low': 1, 'High': 5, 'Poor': 1, 'Good': 5,
               'Quiet': 1, 'Loud': 5, 'Easy': 1, 'Difficult': 5, 2: 2, 3: 3, 4: 4}

        # access breed needed
        top3breeds = top3rec(list(map(dct.get, user_input)))
        # l1 = [breed.replace('_', ' ').replace(' ', '-') for breed in top3breeds]
        # top3_github = list(map(str.lower, l1))
        top3_github = top3breeds

        # print(top3_github)

        # captions
        l2 = [breed.replace('_', ' ').replace('-', ' ') for breed in top3breeds]
        l3 = [re.sub(r'\b[a-z]', lambda m: m.group().upper(), i) for i in l2]
        captions = [breed.replace('And', 'and') for breed in l3]

        # recursively query from our GitHub for our database of images!
        url = "https://api.github.com/repos/{}/{}/git/trees/main?recursive=1".format('dwightvj', 'PIC16B-Project')
        r = requests.get(url)
        res = r.json()

        breed1_files = []
        breed2_files = []
        breed3_files = []
        for file in res["tree"]:
            if "data/dog_photos/{}/".format(top3_github[0]) in file["path"]:
                breed1_files.append(file["path"])
            elif "data/dog_photos/{}/".format(top3_github[1]) in file["path"]:
                breed2_files.append(file["path"])
            elif "data/dog_photos/{}/".format(top3_github[2]) in file["path"]:
                breed3_files.append(file["path"])

        col1, col2, col3 = st.beta_columns(3)

        # format and get the recommended breeds to display across 3 columns
        with col1:
            st.subheader('Top Match')
            with st.spinner('Loading...'):
                url = 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/{}'. \
                    format(random.choice(breed1_files))

                response = requests.get(url)
                img = Image.open(BytesIO(response.content))

                st.image(img)
                st.write("[{0}](https://www.akc.org/dog-breeds/{1}/)".format(captions[0], captions[0].lower().
                                                                             replace(' ', '-')))
        with col2:
            st.subheader('2nd Match')
            with st.spinner('Loading...'):
                url = 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/{}'. \
                    format(random.choice(breed2_files))

                response = requests.get(url)
                img = Image.open(BytesIO(response.content))

                st.image(img)
                st.write("[{0}](https://www.akc.org/dog-breeds/{1}/)".format(captions[1], captions[1].lower().
                                                                             replace(' ', '-')))
        with col3:
            st.subheader('3rd Match')
            with st.spinner('Loading...'):
                url = 'https://raw.githubusercontent.com/dwightvj/PIC16B-Project/main/{}'. \
                    format(random.choice(breed3_files))
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))

                st.image(img)
                st.write("[{0}](https://www.akc.org/dog-breeds/{1}/)".format(captions[2], captions[2].lower().
                                                                             replace(' ', '-')))