import time

import pandas as pd
from PIL import Image
from keras import Sequential
import keras
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import numpy as np
import pandas
import os

pathString = os.path.join("C:", os.sep, "Users", "Noah Parker", "ML_Final_Project_Repo", "dog-breed-identification")
resized_directory = pathString+"\\resized"

resizeWidth = 200
resizeHeight = 200

BATCH_SIZE = 16
NUM_CLASSES = 2
EPOCHS = 6;
img_rows = 200;
img_cols = 200


def trainNClasses(n):
    print("training", n, " classes...")
    X, y = loadLabelsForNClasses(n);
    halfOfXIndex = int(X.shape[0] / 2);
    print("half of X index: ", halfOfXIndex);
    print("size of X: ", X.shape[0], " images.")
    indexRanges = np.arange(0, X.shape[0]);
    np.random.shuffle(indexRanges);

    testingX = X[indexRanges[halfOfXIndex: X.shape[0]]]  # second half of training set
    testingY = y[indexRanges[halfOfXIndex: X.shape[0]]]
    trainingX = X[indexRanges[0: halfOfXIndex]];
    trainingY = y[indexRanges[0: halfOfXIndex]];

#load labels for the N most common breeds
def loadLabelsForNClasses(n):
        d = pandas.read_csv(pathString + '\\labels.csv')
        breeds = d.breed;
        unique = breeds.unique();  # All of the classes (120)
        print("Loading labels for ", n, " classes.")

        top_N = list(d['breed'].value_counts()[0:n].index);
        print("most common breeds:", top_N)

        top_N_breeds = d[d.breed.isin(top_N)];
        print("top N: ", top_N)
        imageRows = None;
        yRows = None;
        for i in range(n):
            # image_count = most_common[unique[i]]

            print(" ", top_N[i], "selected.");
            rows = d[d.breed == top_N[i]].copy();
            images = rows;
            rows.breed[rows.index] = i;
            print("", rows.id.size, " images for ", top_N[i])
            if imageRows is None:
                imageRows = rows.id
                yRows = np.array(rows.breed, dtype=np.int32);
            else:
                print("appending images! size before: ", imageRows.size, ", size of rows: ", rows.id.size)
                imageRows = pandas.concat((imageRows, rows.id));
                print("size after: ", imageRows.size)
                yRows = np.concatenate((yRows, np.array(rows.breed, dtype=np.int32)));
            print("size of imageRows: ", imageRows.size)

        training_labels = yRows;
        return reshapeTrainingImagesKeras(imageRows), yRows;


def reshapeTrainingImagesKeras(fileNames):
    # X should be n_samples x n_features according to scikit learn.
    X = np.empty((fileNames.shape[0], resizeWidth, resizeHeight))

    count = 0;
    for imageName in fileNames:
        try:
            with Image.open(os.path.join(resized_directory, imageName + ".jpg")) as im:
                array = np.array(im);
                # if count == 0 or count == 90:
                # im.show()
                reshaped = np.reshape(array, (resizeWidth, resizeHeight))
                X[count] = reshaped;
                count = count + 1;
        except IOError as e:
            print("Error loading resized image!.... ", e);

    return X


x, y = loadLabelsForNClasses(2);


# print("shape of dog X: ", x.shape);
# print("shape of dog Y: ", y.shape);
# print("y pre-reshape: ", y)
threeQuarters = int(x.shape[0] * 3/4);
x_train = x[0:threeQuarters];
y_train = y[0:threeQuarters];
x_test = x[threeQuarters : x.shape[0] ]
y_test = y[threeQuarters : y.shape[0] ]

x_train = x_train.reshape(x_train.shape[0],  200, 200, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

INPUT_SHAPE = (img_rows, img_cols, 1)
#
# #
# #
# # print("AFTER RESHAPE: ")
# # print("x_train.shape; ", x_train.shape);
# # x_train = x_train.astype('float32')
# # x_test = x_test.astype('float32')
# # x_train /= 255;
# # x_test /= 255;
# #
# # #this converts to one-hot encoding which is nice.
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES);
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES);
#print("y_train one hot encoding: ", y_train);

#*************************************************************************************
model = Sequential();
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=INPUT_SHAPE));
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#
# #last layer should spit out probabilities
model.add(Dense(NUM_CLASSES, activation='softmax'))
#
# print("Compiling the model...")
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='sgd',
              metrics=['accuracy'])
# #
# # print("Done!")
# # print("Training over ", EPOCHS, "epochs...")
# # startTime = time.time();
#
#
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,#)
          validation_data=(x_test, y_test));
# #
# timeAfter = time.time();
# duration = timeAfter - startTime;
# print("Done training! ", EPOCHS," took ", duration/60, "minutes.")
#
# score = model.evaluate(x_test, y_test, verbose=0)
# print("Test loss: ", score[0])
# print("Test accuracy: ", score[1])
#
# print("hello!")


