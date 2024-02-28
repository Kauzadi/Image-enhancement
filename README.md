# Image-enhancement
Here’s using convolution neural network and training it using deep learning frameworks TensorFlow and Keras to enhance low-resolution MRI images to better resolution ones. 
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

image_height = 128
image_width = 128
num_channels = 3

train_images_lowres = np.random.rand(100, image_height, image_width, num_channels)
train_images_highres = np.random.rand(100, image_height, image_width, num_channels)

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(image_height, image_width, num_channels)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(1, (3, 3), activation='relu', padding='same'))
    return model

model = build_model()
model.compile(optimizer='adam', loss='mse')

model.fit(train_images_lowres, train_images_highres, epochs=10, batch_size=32, validation_split=0.2)

test_images_lowres = np.random.rand(10, image_height, image_width, num_channels)

enhanced_images = model.predict(test_images_lowres)
