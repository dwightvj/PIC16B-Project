import streamlit as st

def main():
    # title
    st.header("Model Architecture")
    st.write("\n")

    # create polished layout of 3 columns in order to center the png below
    col1, col2, col3 = st.beta_columns([1, 6, 1])

    with col1:
        st.write("")

    with col2:
        # center  our model_architecture.png in the page
        st.image('model_architecture.png', width=500, caption='Convolutional Neural Network')

    with col3:
        st.write("")
