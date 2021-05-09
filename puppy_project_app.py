# -*- coding: utf-8 -*-
import streamlit as st

def main():
    
    st.title("Puppy Party")
    
    image = None
    
    image = st.file_uploader("Upload an image of a dog", ["png", "jpg", "jpeg", "jfif", "heic"])
    

if __name__ == "__main__":
    main()

    
    
    