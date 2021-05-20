import streamlit as st

def main():
    st.header("Give Us Feedback!")
    # st.markdown("""
    #       <iframe src="https://formfacade.com/headless/101215250839582918673/home/form/1FAIpQLSdT9Wpq4pQ28nc1nSq5NcOaClCm25tzP6AizNrZVWeHcBEMYQ" width="640" height="457" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>
    #     """, unsafe_allow_html=True)

    col1, col2, col3 = st.beta_columns([1, 6, 1])

    with col1:
        st.write("")

    with col2:
        st.markdown("""
                  <iframe src="https://formfacade.com/headless/101215250839582918673/home/form/1FAIpQLSdT9Wpq4pQ28nc1nSq5NcOaClCm25tzP6AizNrZVWeHcBEMYQ" width="640" height="385" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>
                """, unsafe_allow_html=True)

    with col3:
        st.write("")



