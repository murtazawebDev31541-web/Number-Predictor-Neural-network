import streamlit as st
import requests
from PIL import Image

# Exact URL where FastAPI is running
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="MNIST Digit Predictor", page_icon="🔢")

st.title("🔢 MNIST Handwritten Digit Predictor")
st.write("Upload an image of a single handwritten digit (0–9) to make a prediction.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Digit Image", width=200)

    if st.button("Predict Digit"):
        with st.spinner("Analyzing image..."):
            try:
                # Send the file to FastAPI
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"### Predicted Digit: **{result['prediction']}**")
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI server. Please check if `uvicorn main:app --reload` is running in your terminal.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")