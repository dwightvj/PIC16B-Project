import streamlit as st

def main():
    st.header("Model Architecture")
    st.write("\n")
    #st.image('model_architecture.png', width=550, caption='Convolutional Neural Network')

    col1, col2, col3 = st.beta_columns([1, 6, 1])

    with col1:
        st.write("")

    with col2:
        st.image('model_architecture.png', width=500, caption='Convolutional Neural Network')

    with col3:
        st.write("")
