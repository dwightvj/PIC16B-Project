import first_app, second_app, third_app, fourth_app
import streamlit as st


Pages = {"Upload Your Own Image": first_app, "Use A Sample Image": third_app, "Model Architecture": second_app,
         'Find Your Perfect Dog!': fourth_app}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(Pages.keys()))

Pages[selection].main()
