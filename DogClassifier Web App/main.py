import first_app, second_app, third_app, fourth_app
import streamlit as st


Pages = {"Predict Dog Breed": first_app, "Model Architecture": second_app, "Give Us Feedback!": third_app,
         'Find Your Perfect Dog': fourth_app}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(Pages.keys()))

Pages[selection].main()
