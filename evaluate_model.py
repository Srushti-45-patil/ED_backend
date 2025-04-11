import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("emotion_model.h5")

# Load the test dataset generator (use the same preprocessing)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_data_dir = r"C:\Users\reals\ML_Project_Backend\Datasets\Processed_FER\test"  # Update this path if needed

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"🧪 Test Accuracy: {test_acc:.4f}")
print(f"📉 Test Loss: {test_loss:.4f}")
