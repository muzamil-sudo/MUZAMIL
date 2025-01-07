import streamlit as st
video_input = st.camera_input("Capture Image")

if video_input:
    st.image(video_input, caption="Captured Image", use_column_width=True)
