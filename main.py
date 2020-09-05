import pandas as pd
import numpy as np

from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical

# Read CSV File
data = pd.read_csv(
    "./fer2013.csv")
data.head()

data["Usage"].value_counts()

# Split data into training, public_test, private_test
training = data.loc[data["Usage"] == "Training"]
public_test = data.loc[data["Usage"] == "PublicTest"]
private_test = data.loc[data["Usage"] == "PrivateTest"]

# Preprocessing

train_labels = training["emotion"]
train_labels = to_categorical(train_labels)

train_pixels = training["pixels"].str.split(" ").tolist()
train_pixels = np.uint8(train_pixels)
train_pixels = train_pixels.reshape((28709, 48, 48, 1))
train_pixels = train_pixels.astype("float32") / 255

private_labels = private_test["emotion"]
private_labels = to_categorical(private_labels)
private_pixels = private_test["pixels"].str.split(" ").tolist()
private_pixels = np.uint8(private_pixels)
private_pixels = private_pixels.reshape((3589, 48, 48, 1))
private_pixels = private_pixels.astype("float32") / 255

public_labels = public_test["emotion"]
public_labels = to_categorical(public_labels)
public_pixels = public_test["pixels"].str.split(" ").tolist()
public_pixels = np.uint8(public_pixels)
public_pixels = public_pixels.reshape((3589, 48, 48, 1))
public_pixels = public_pixels.astype("float32") / 255


# Build model:
# I am using the VGG16 model

model = models.Sequential()
model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
model.add(layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(7, activation='softmax'))

# print(summary)
model.summary()

# Compile the model using Adam optimizer
model.compile(optimizer="Adam", loss="categorical_crossentropy",
              metrics=["accuracy"])

# Fit the model
model.fit(train_pixels, train_labels, batch_size=256,
          epochs=30, validation_data=(private_pixels, private_labels))
model.save('./trained_model.h5')
