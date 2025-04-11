import os
import cv2
import numpy as np

dataset_path = r"C:\Users\reals\ML_Project_Backend\Datasets\FER-2013"
output_path = r"C:\Users\reals\ML_Project_Backend\Datasets\Processed_FER"

if not os.path.exists(output_path):
    os.makedirs(output_path)

print("Starting preprocessing...")  # Debugging

# Loop through training and test sets
for dataset_type in ["train", "test"]:
    dataset_folder = os.path.join(dataset_path, dataset_type)
    output_dataset_folder = os.path.join(output_path, dataset_type)

    if not os.path.exists(output_dataset_folder):
        os.makedirs(output_dataset_folder)

    # Loop through emotion categories
    for emotion in os.listdir(dataset_folder):  
        emotion_folder = os.path.join(dataset_folder, emotion)
        output_emotion_folder = os.path.join(output_dataset_folder, emotion)

        if not os.path.exists(output_emotion_folder):
            os.makedirs(output_emotion_folder)

        # Loop through image files inside emotion folder
        for img_name in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_name)

            # Read image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Skipping {img_name}, unable to read")  # Debugging
                continue

            # Resize image to 48x48 (standard for FER-2013)
            img = cv2.resize(img, (48, 48))

            # Save preprocessed image
            save_path = os.path.join(output_emotion_folder, img_name)
            cv2.imwrite(save_path, img)

            print(f"✅ Processed: {img_name}")  # Debugging

print("🎉 Preprocessing completed successfully!")
