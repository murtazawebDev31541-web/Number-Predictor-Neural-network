from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, ImageOps
from pathlib import Path
import numpy as np
import pickle
import io

app = FastAPI(title="MNIST Digit Prediction API")

# Safely get directory path (works in both Colab and local Python scripts)
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# Load the trained model
MODEL_PATH = BASE_DIR / "mnist_model.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    model = None
    print(f"Warning: Could not load model file from {MODEL_PATH}: {e}")


@app.get("/")
def home():
    return {"message": "MNIST API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded. Ensure mnist_model.pkl is in the project folder."
        )

    try:
        # 1. Read uploaded image file
        image_bytes = await file.read()

        # 2. Open image and convert to Grayscale
        image = Image.open(io.BytesIO(image_bytes)).convert("L")

        # 3. Auto-invert colors if background is light/white (MNIST expects black background)
        if np.mean(image) > 127:
            image = ImageOps.invert(image)

        # 4. Resize to 28x28 pixels
        image = image.resize((28, 28))

        # 5. Convert to float array and normalize (0 to 1)
        image_array = np.array(image, dtype="float32") / 255.0

        # 6. Flatten array from (28, 28) to (1, 784)
        image_array = image_array.reshape(1, 784)

        # 7. Make prediction
        prediction = model.predict(image_array)

        # 8. Handle both Keras 2D arrays and Scikit-Learn 1D arrays
        if np.ndim(prediction) == 2 and prediction.shape[1] > 1:
            digit = int(np.argmax(prediction[0]))
        else:
            digit = int(prediction[0])

        return {"prediction": digit}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )