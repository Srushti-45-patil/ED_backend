import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load preprocessed data
if not (os.path.exists("X_train_facial.npy") and os.path.exists("X_train_audio.npy")):
    raise FileNotFoundError("Preprocessed data files not found. Run preprocessing first.")

X_train_facial = np.load("X_train_facial.npy")
X_test_facial = np.load("X_test_facial.npy")
X_train_audio = np.load("X_train_audio.npy")
X_test_audio = np.load("X_test_audio.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Just in case the labels aren't already one-hot encoded
if y_train.ndim == 1 or y_train.shape[1] == 1:
    y_train = to_categorical(y_train, num_classes=8)
    y_test = to_categorical(y_test, num_classes=8)

# Define input layers
facial_input = Input(shape=(48, 48, 1), name="facial_input")
audio_input = Input(shape=(40, 1), name="audio_input")

# Facial branch
x1 = Flatten()(facial_input)
x1 = Dense(128, activation="relu")(x1)
x1 = Dense(64, activation="relu")(x1)

# Audio branch
x2 = Flatten()(audio_input)
x2 = Dense(128, activation="relu")(x2)
x2 = Dense(64, activation="relu")(x2)

# Combine branches
merged = Concatenate()([x1, x2])
out = Dense(8, activation="softmax", name="output_layer")(merged)

# Create model
model = Model(inputs=[facial_input, audio_input], outputs=out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
print("🚀 Training started...")
model.fit([X_train_facial, X_train_audio], y_train, 
          validation_data=([X_test_facial, X_test_audio], y_test), 
          epochs=25, batch_size=32)

# Save model
model.save("multimodal_emotion_model.h5")
print("✅ Model trained and saved as 'multimodal_emotion_model.h5'")
