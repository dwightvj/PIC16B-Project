# -*- coding: utf-8 -*-
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# displays the uploaded image to the screen
def display_img(image):
    st.balloons() # display the balloons
    dis_img = Image.open(image)
    st.image(dis_img, width = 500)

def main():
    
    st.title("Puppy Party")
    
    image = None
    
    image = st.file_uploader("Upload an image of a dog", ["png", "jpg", "jpeg", "jfif", "heic"])
    
    if image is not None:
        display_img(image)
    
if __name__ == "__main__":
    main()

    
    
    
