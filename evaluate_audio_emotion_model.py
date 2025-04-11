import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Load the test dataset
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Reshape X_test to match model input (add time_step=1)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Load the trained model
model = tf.keras.models.load_model("audio_emotion_model.h5")

# Get model predictions
y_pred_probs = model.predict(X_test)  # Get probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert to class labels
y_true = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels to class labels

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
