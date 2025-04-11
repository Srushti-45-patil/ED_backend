import os

os.rename("X_train.npy", "X_train_audio.npy")
os.rename("X_test.npy", "X_test_audio.npy")

print("✅ Audio feature files renamed successfully!")
