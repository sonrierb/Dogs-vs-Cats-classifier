import streamlit as st
import requests

st.title("Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        
        # Prepare file for API
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }

        # Call FastAPI endpoint
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files=files
        )

        if response.status_code == 200:
            result = response.json()

            st.success(f"Prediction: **{result['label']}**")
            st.info(f"Confidence: **{round(result['confidence'], 3)}**")
        else:
            st.error("Something went wrong. Check FastAPI server.")