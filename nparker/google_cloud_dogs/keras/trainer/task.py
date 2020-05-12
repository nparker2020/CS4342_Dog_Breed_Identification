import time

#import pandas as pd
#from PIL import Image, ImageOps
#import keras
import keras
from keras.models import Sequential
#from tensorflow.python.keras.models import Sequential

from keras.datasets import mnist
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, ZeroPadding2D
#from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, ZeroPadding2D
import numpy as np
#import pandas
import os
from keras import optimizers
#import skimage.transform as tf
import io
from tensorflow.python.lib.io import file_io
from datetime import datetime


pathString = os.path.join("C:", os.sep, "Users", "Noah Parker", "ML_Final_Project_Repo", "dog-breed-identification")
resized_directory = pathString+"\\resized"

resizeWidth = 200
resizeHeight = 200

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 1500;
img_rows = 100
img_cols = 100
MODEL_NAME = 'model_normalstride_e1500';


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

        top_N = list(d['breed'].value_counts()[0:n].index);
      
        top_N_breeds = d[d.breed.isin(top_N)];
        
        imageRows = None;
        yRows = None;
        for i in range(n):

            print(" ", top_N[i], "selected.");
            rows = d[d.breed == top_N[i]].copy();
            images = rows;
            rows.breed[rows.index] = i;
            print("", rows.id.size, " images for ", top_N[i])
            if imageRows is None:
                imageRows = rows.id
                yRows = np.array(rows.breed, dtype=np.int32);
            else:
                imageRows = pandas.concat((imageRows, rows.id));
                yRows = np.concatenate((yRows, np.array(rows.breed, dtype=np.int32)));
            

        training_labels = yRows;
        return reshapeTrainingImagesKeras(imageRows, yRows)#, yRows;

def duplicateImages(id_list, yRows):
    count = 0
    X = 0
    Y = 0
    Z = 0
    W = 0
    flipped = True
    rotate = False
    translate = True
    originalLabels = np.zeros(id_list.shape[0]);
    flippedLabels = np.zeros(id_list.shape[0]);
    translateLabels = np.zeros(id_list.shape[0]);
    ogCount = 0
    flipCount = 0;
    translateLabels = 0;
    print("duplicating data...")
    for image_file in id_list:
        label = yRows[count];
        
        try:
            temp_img = Image.open(os.path.join(resized_directory, image_file+".jpg"))
            
            temp_img = np.asarray(temp_img)
            if flipped:
                flip_img = np.fliplr(temp_img)
            if rotate:
                rot_img = tf.rotate(temp_img, 7)
            if translate:
                x, y = 10, 10
                trans_locs = [(x, y), (-x, y), (-x, -y), (x, -y), (x, 0), (-x, 0), (0, y), (0, -y)]
                trans = np.random.randint(0, len(trans_locs), 1)[0]
                trans_mx = tf.EuclideanTransform(translation=trans_locs[trans])
                trans_img = tf.warp(temp_img, trans_mx)

            temp_img = temp_img.reshape((-1, 1))
            if flipped:
                flip_img = flip_img.reshape((-1, 1))
            if translate:
                trans_img = trans_img.reshape((-1, 1))
            if rotate:
                rot_img = rot_img.reshape((-1, 1))

            if count == 0:
                X = temp_img
                originalLabels[ogCount] = label;
                ogCount = ogCount + 1;
                if flipped:
                    Y = flip_img
                if translate:
                    Z = trans_img
                if rotate:
                    W = rot_img
            else:
                X = np.append(X, temp_img, axis=1)
                originalLabels[ogCount] = label;
                ogCount = ogCount + 1;
                if flipped:
                    Y = np.append(Y, flip_img, axis=1)
                if translate:
                    Z = np.append(Z, trans_img, axis=1)
                if rotate:
                    W = np.append(W, rot_img, axis=1)
        except:
            print("error opening image: " + str(image_file))

        if count % 1000 == 0:
            print(count)
        count += 1

    if flipped:
        print("Y.T shape:", Y.T.shape)
    if translate:
        print("Z.T shape:", Z.T.shape)
    if rotate:
        print("W.T shape:", W.T.shape)

    comb = np.hstack((X, Y, Z))
    print("comb.T.shape: ", comb.T.shape)
    concattedLabels = np.hstack((originalLabels, originalLabels, originalLabels))
    print("concattedLabels: ", concattedLabels.shape)
    np.save('dog_breed_og_flp_trans_data.npy', comb.T)
    np.save('dog_breed_og_flp_trans_labels.npy', concattedLabels)
    return comb, concattedLabels

def reshapeTrainingImagesKeras(fileNames, labels):
    # X should be n_samples x n_features according to scikit learn.
    X = np.empty((fileNames.shape[0], resizeWidth, resizeHeight))

    images, labels = duplicateImages(fileNames, labels);

    return images, labels

def loadNClassesFromFile(NUM_CLASSES):
    x = np.load('dog_breed_og_flp_trans_data.npy')
    y = np.load('dog_breed_og_flp_trans_labels.npy')

    x = x.T;
    return x, y


def loadNClassesFromGCP(NUM_CLASSES):
    input_data = np.load(io.BytesIO(file_io.read_file_to_string('gs://ml-final-project-bucket/data/dog_breed_og_flp_trans_data_color_rotated.npy', binary_mode=True)))
    input_labels = np.load(io.BytesIO(file_io.read_file_to_string('gs://ml-final-project-bucket/data/dog_breed_og_flp_trans_labels_color_rotated.npy', binary_mode=True)))
    x = input_data.T;
    y = input_labels;
    return x, y

def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as fp:
            fp.write(input_f.read())
            
def scheduler(epoch, lr):
    learning_rate = 0.0;
    if epoch < 300:
        learning_rate = lr;
    elif epoch >= 300 and epoch < 1000:
        learning_rate = .000001;
    elif epoch >= 1000:
        learning_rate = .0000001;
    print("*****Learning_rate this epoch: ", learning_rate, " *****");
    return learning_rate;

#x, y = loadNClassesFromFile(NUM_CLASSES);
x,y = loadNClassesFromGCP(NUM_CLASSES);


indices = np.arange(0, x.T.shape[0]);
np.random.shuffle(indices);

x = x.T;

threeQuarters = int(x.shape[0] * 7/8);
x_train = x[indices[0:threeQuarters]];
y_train = y[indices[0:threeQuarters]];
x_test = x[indices[threeQuarters : x.shape[0] ]]
y_test = y[indices[threeQuarters : y.shape[0]]]

x_train = x_train.reshape(x_train.shape[0],  img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255;
x_test /= 255;


INPUT_SHAPE = (img_rows, img_cols, 3)

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES);
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES);

model = Sequential();
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=INPUT_SHAPE));
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.4));
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

# last layer should spit out probabilities for each class
model.add(Dense(NUM_CLASSES, activation='softmax'))

#beginning learning rate
learning_rate = .00001
print("Compiling the model. Learning rate: ", learning_rate, "epochs: ", EPOCHS)

optimizer = keras.optimizers.Adam(lr=learning_rate);

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

print("Done!")
print("*******************************************************************************************")
print(model.summary())

print("Training over ", EPOCHS, "epochs...")
startTime = time.time();

#callback for degrading learning rate scheduler
callback = keras.callbacks.LearningRateScheduler(scheduler)
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          callbacks=[callback],
          verbose=1,#)
          validation_data=(x_test, y_test));

timeAfter = time.time();
duration = timeAfter - startTime;
print("Done training! ", EPOCHS," took ", duration/60, "minutes.")

print("Saving model...")
MODEL_NAME += str(datetime.now()) + '.hdf5'
MODEL_NAME = MODEL_NAME.replace(":", "_")
model.save(MODEL_NAME)

#copy model export to GCS bucket
print("Copying ", MODEL_NAME, " to GCS bucket...")
copy_file_to_gcs('gs://ml-final-project-bucket/models/', MODEL_NAME)
print("Copy Complete!");



