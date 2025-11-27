from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = FastAPI(title="Cat vs Dog Classifier API")

# -------------------------
# Load model ONLY ONCE
# -------------------------
model = load_model("saved_model/model.h5")

# -------------------------
# Predict Function
# -------------------------
def predict_image(img):
    img = cv2.resize(img, (100, 100))      # your model input size
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred < 0.5:
        return "Cat", 1 - pred
    else:
        return "Dog", pred

# -------------------------
# API Endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)

    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Invalid image file"}, status_code=400)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    label, confidence = predict_image(img)

    return {
        "label": label,
        "confidence": float(confidence)
    }

