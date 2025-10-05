"""
dataset_generator.py

Regenerates the small synthetic dataset used by this project.
"""
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import os

BASE = Path(__file__).parent
TRAIN_HEALTHY = BASE / "dataset" / "train" / "healthy"
TRAIN_DISEASED = BASE / "dataset" / "train" / "diseased"
TEST_HEALTHY = BASE / "dataset" / "test" / "healthy"
TEST_DISEASED = BASE / "dataset" / "test" / "diseased"

def create_synthetic_leaf(save_path, kind="healthy", size=(64,64), seed=0):
    rng = np.random.RandomState(seed)
    img = Image.new("RGB", size, (34, 139, 34))
    draw = ImageDraw.Draw(img)
    bbox = [size[0]*0.1, size[1]*0.15, size[0]*0.9, size[1]*0.85]
    draw.ellipse(bbox, fill=(50,160,70))
    if kind == "diseased":
        for _ in range(6):
            x = rng.randint(size[0]*0.15, size[0]*0.85)
            y = rng.randint(size[1]*0.2, size[1]*0.8)
            r = rng.randint(3,9)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(30,30,30))
    px = img.load()
    for _ in range(80):
        x = rng.randint(0, size[0]-1)
        y = rng.randint(0, size[1]-1)
        px[x,y] = tuple(map(lambda v: max(0, min(255, int(v + rng.randint(-10,10)))), px[x,y]))
    img.save(save_path)

def regenerate():
    # create small dataset
    num_train = 80
    num_test = 20
    TRAIN_HEALTHY.mkdir(parents=True, exist_ok=True)
    TRAIN_DISEASED.mkdir(parents=True, exist_ok=True)
    TEST_HEALTHY.mkdir(parents=True, exist_ok=True)
    TEST_DISEASED.mkdir(parents=True, exist_ok=True)
    for i in range(num_train):
        create_synthetic_leaf(TRAIN_HEALTHY / f"h_{i:03d}.png", "healthy", seed=i)
        create_synthetic_leaf(TRAIN_DISEASED / f"d_{i:03d}.png", "diseased", seed=1000+i)
    for i in range(num_test):
        create_synthetic_leaf(TEST_HEALTHY / f"h_{i:03d}.png", "healthy", seed=2000+i)
        create_synthetic_leaf(TEST_DISEASED / f"d_{i:03d}.png", "diseased", seed=3000+i)

if __name__ == "__main__":
    regenerate()
