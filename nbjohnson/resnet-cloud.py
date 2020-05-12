from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
import datetime
from sklearn.model_selection import train_test_split
import numpy as np
from google.cloud import storage
import logging
import io
from tensorflow.python.lib.io import file_io

input_data = np.load(io.BytesIO(file_io.read_file_to_string('gs://machine-learning-final-project/data/dog_breed_og_flp_trans_data_test_comb.npy', binary_mode=True)))
logging.info(input_data.shape)
input_labels = np.load(io.BytesIO(file_io.read_file_to_string('gs://machine-learning-final-project/data/dog_breed_og_flp_trans_labels_test_comb.npy', binary_mode=True)))

X_train, X_val, y_train, y_val = train_test_split(input_data, input_labels, test_size=0.2, random_state=42)
logging.info("done splitting")


new_input = Input(shape=(224,224,3))  # should be 224,224

# graph changes over epochs
logdir = "gs://machine-learning-project/logs2/scalars/" + datetime.time().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)

model = Sequential()
model.add(ResNet50(weights='imagenet', include_top=False, input_tensor=new_input, pooling='avg'))
model.trainable = False
model.add(Flatten())
model.add(Dense(120, activation='softmax'))
logging.info(model.summary())


model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False),
                  metrics=['accuracy', 'categorical_crossentropy', 'mae'])

model.fit(input_data, input_labels, epochs=10000, batch_size=512, validation_data=(X_val, y_val),
              callbacks=[tensorboard_callback])

model.save("gs://machine-larning-project/model-output/test-resnet-model", save_format='tf')

# for later
# model = load_model("test-resnet-model")