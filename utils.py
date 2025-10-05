"""
utils.py

Image utilities used by the project.
"""
from PIL import Image
import numpy as np

def load_and_preprocess(path, size=(64,64)):
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.array(img)/255.0
    return arr
