import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model(
    "plant_disease_recog_model_pwp.keras",
    compile=False
)

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimization (size reduction)import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model(
    "plant_disease_recog_model_pwp.keras",
    compile=False
)

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimization (size reduction)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open("plant_disease_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted successfully to TFLite")

converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open("plant_disease_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted successfully to TFLite")
