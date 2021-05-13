import first_app
import streamlit as st


Pages = {"Detect My Dog": first_app}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(Pages.keys()))

Pages[selection].main()
