"""
predict.py

Usage:
    python predict.py path/to/image.png

Loads model.h5 and prints predicted class (healthy/diseased) with probability.
"""
import sys
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model

LABELS = {0: "healthy", 1: "diseased"}

def preprocess_image(path, target_size=(64,64)):
    img = Image.open(path).convert("RGB").resize(target_size)
    arr = np.array(img)/255.0
    arr = arr.reshape((1,)+arr.shape)
    return arr

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.png")
        return
    image_path = sys.argv[1]
    model_path = os.path.join(os.path.dirname(__file__), "model.h5")
    model = load_model(model_path)
    x = preprocess_image(image_path)
    p = model.predict(x)[0][0]
    cls = 1 if p >= 0.5 else 0
    print(f"Predicted: {LABELS[cls]} (prob={p:.3f})")

if __name__ == "__main__":
    main()
