import streamlit as st
from PIL import Image, ImageOps
from pathlib import Path
import numpy as np
import pickle
import io


# --------------------------------------------------
# Load FastAPI / Model Logic
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mnist_model.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

except Exception as e:
    model = None
    model_error = str(e)


# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------

def predict_digit(image_bytes):

    if model is None:
        raise Exception(
            f"Model is not loaded. Ensure mnist_model.pkl "
            f"is in the project folder. Error: {model_error}"
        )

    # 1. Read uploaded image file
    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("L")

    # 2. Auto-invert colors if background is light/white
    # MNIST expects black background
    if np.mean(image) > 127:
        image = ImageOps.invert(image)

    # 3. Resize to 28x28 pixels
    image = image.resize((28, 28))

    # 4. Convert to float array and normalize
    image_array = np.array(
        image,
        dtype="float32"
    ) / 255.0

    # 5. Flatten array from (28, 28) to (1, 784)
    image_array = image_array.reshape(1, 784)

    # 6. Make prediction
    prediction = model.predict(image_array)

    # 7. Handle both Keras 2D arrays
    # and Scikit-Learn 1D arrays
    if (
        np.ndim(prediction) == 2
        and prediction.shape[1] > 1
    ):
        digit = int(
            np.argmax(prediction[0])
        )
    else:
        digit = int(prediction[0])

    return digit


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(
    page_title="MNIST Digit Predictor",
    page_icon="🔢"
)

st.title("🔢 MNIST Handwritten Digit Predictor")

st.write(
    "Upload an image of a single handwritten digit "
    "(0–9) to make a prediction."
)


uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Digit Image",
        width=200
    )

    if st.button("Predict Digit"):

        with st.spinner("Analyzing image..."):

            try:

                # Get uploaded image bytes
                image_bytes = uploaded_file.getvalue()

                # Run prediction
                digit = predict_digit(
                    image_bytes
                )

                # Display result
                st.success(
                    f"### Predicted Digit: **{digit}**"
                )

            except Exception as e:

                st.error(
                    f"An unexpected error occurred: {e}"
                )