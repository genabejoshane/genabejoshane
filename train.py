"""
train.py

Small script to train a tiny CNN on images in dataset/train and dataset/test.
Saves the trained model to model.h5 in the project root.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

BASE_DIR = os.path.dirname(__file__)
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

def build_model(input_shape=(64,64,3)):
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(64,64),
        batch_size=16,
        class_mode='binary'
    )
    test_gen = train_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(64,64),
        batch_size=16,
        class_mode='binary'
    )
    model = build_model()
    model.fit(train_gen, epochs=6, validation_data=test_gen)
    model.save(MODEL_PATH)
    print("Saved model to", MODEL_PATH)

if __name__ == "__main__":
    main()
