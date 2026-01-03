import tensorflow as tf

IMG_SIZE = (160, 160)
BATCH = 32

train = tf.keras.utils.image_dataset_from_directory(
    "../dataset/plantdisease",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    validation_split=0.2,
    subset="training",
    seed=123
)

val = tf.keras.utils.image_dataset_from_directory(
    "../dataset/plantdisease",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    validation_split=0.2,
    subset="validation",
    seed=123
)

NUM_CLASSES = len(train.class_names)
print("Classes:", train.class_names)

norm = tf.keras.layers.Rescaling(1./255)
train = train.map(lambda x, y: (norm(x), y))
val = val.map(lambda x, y: (norm(x), y))

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(160,160,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train, validation_data=val, epochs=10)

model.save("../models/plant_disease_recog_model_pwp.keras")
print("✅ Model saved successfully")
