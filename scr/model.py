import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from PIL import Image

# using a pre-trained model
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
base_model.trainable = False  # Freeze the base model

# Add custom layers on top for our specific task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(
    'data/delivery/',  # Path to training data
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Train the model
model.fit(train_generator, epochs=10)

#save the model
model.save('mobilenet_custom_model.h5')