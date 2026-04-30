import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

train_dir = 'Fruit classifier/train'

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2 # 20% of images for validation
)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=4,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=4,
        class_mode='categorical',
        subset='validation')

base_model = tf.keras.applications.MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Starting training...")
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator
)
print("Training completed.")

model.save('fruit_classifier_model.keras')
print("Model saved to 'fruit_classifier_model.keras'")

test_apple_path = 'Fruit classifier/test/test_apple.jpg'
test_mango_path = 'Fruit classifier/test/test_mango.jpg'

class_labels = list(train_generator.class_indices.keys())

def predict_image(image_path):
    print(f"\nPredicting image: {image_path}")
    img = keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    prediction = model.predict(img_array)

    predicted_idx = np.argmax(prediction[0])
    predicted_class = class_labels[predicted_idx]
    score = prediction[0][predicted_idx]

    print(f"Prediction result: {predicted_class} (Confidence: {score:.4f})")

predict_image(test_apple_path)
predict_image(test_mango_path)
