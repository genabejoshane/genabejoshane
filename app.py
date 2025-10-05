"""
app.py

A minimal Flask API to upload an image and get prediction.
Run: python app.py
"""
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import os
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
MODEL = load_model(os.path.join(os.path.dirname(__file__), "model.h5"))
LABELS = {0: "healthy", 1: "diseased"}

def preprocess_bytes(image_bytes, target_size=(64,64)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(target_size)
    arr = np.array(img)/255.0
    arr = arr.reshape((1,)+arr.shape)
    return arr

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image provided"}), 400
    f = request.files["image"]
    img_bytes = f.read()
    x = preprocess_bytes(img_bytes)
    p = float(MODEL.predict(x)[0][0])
    cls = 1 if p >= 0.5 else 0
    return jsonify({"class": LABELS[cls], "probability": p})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
