import os
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Path to your dataset folder (modify if needed)
DATASET_PATH = r"C:\Users\reals\ML_Project_Backend\Datasets\RAVDESS"

# Emotion mapping based on RAVDESS naming convention
emotion_labels = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Function to extract MFCC features
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)  # Take mean of MFCCs over time
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Lists to store features and labels
X, y = [], []

# Loop through actor subfolders
for actor in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor)
    if os.path.isdir(actor_path):  # Ensure it's a directory
        for file in os.listdir(actor_path):
            if file.lower().endswith(".wav"):  # Ensure it's a .wav file
                file_path = os.path.join(actor_path, file)
                emotion_code = file.split("-")[2]  # Extract emotion code
                if emotion_code in emotion_labels:
                    features = extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(int(emotion_code) - 1)  # Convert to 0-based index
                    else:
                        print(f"Skipping file: {file_path}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# One-hot encode labels
y = to_categorical(y, num_classes=8)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed data
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print(f"✅ Preprocessing complete! {X_train.shape[0]} train samples, {X_test.shape[0]} test samples saved.")
