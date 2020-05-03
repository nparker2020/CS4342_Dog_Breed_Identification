import time

import numpy as no
import pandas as pd
from keras import Sequential
import keras
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

BATCH_SIZE = 64
NUM_CLASSES = 10
EPOCHS = 6;
img_rows = 28;
img_cols = 28

(x_train, y_train), (x_test, y_test) = mnist.load_data();
print("BEFORE RESHAPE:")
print("x_train.shape; ", x_train.shape);
print("y_train.shape: ", y_train.shape);
print("y_train: ")
print(y_train)

#This asks which format our backend needs (i.e. tensorflow, etc.)
# if K.image_data_format() == 'channels_first':
#     print("image data format is channels first!")
# else:
#     print(K.image_data_format())
x_train = x_train.reshape(x_train.shape[0],  img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
INPUT_SHAPE = (img_rows, img_cols, 1)



print("AFTER RESHAPE: ")
print("x_train.shape; ", x_train.shape);
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255;
x_test /= 255;

#this converts to one-hot encoding which is nice.
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES);
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES);
print("y_train reformated: ")
print("shape: ", y_train.shape);
print(y_train)

model = Sequential();
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=INPUT_SHAPE));
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

#last layer should spit out probabilities
model.add(Dense(NUM_CLASSES, activation='softmax'))

print("Compiling the model...")
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='sgd',
              metrics=['accuracy'])

print("Done!")
print("Training over ", EPOCHS, "epochs...")
startTime = time.time();
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(x_test, y_test))

timeAfter = time.time();
duration = timeAfter - startTime;
print("Done training! ", EPOCHS," took ", duration/60, "minutes.")

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

print("hello!")