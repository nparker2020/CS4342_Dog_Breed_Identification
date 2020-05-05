from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import datetime
import numpy as np
from sklearn.model_selection import train_test_split

num_layers = 3

data_train = []
labels_train = []
data_validate = []
labels_validate = []

data_in = np.load("dog_breed_og_flp_trans_data.npy")
print("data")
print(data_in.shape)

labels_in = np.load("dog_breed_og_flp_trans_labels.npy")
print("labels")
print(labels_in.shape)

test_in = np.load("dog_breed_test_data.npy")
print("test")
print(test_in.shape)
print("Done loading data")

X_train, X_val, y_train, y_val = train_test_split(data_in, labels_in, test_size=0.2, random_state=42)
print("done splitting")

# graph changes over epochs
logdir = "logs/scalars/" + datetime.time().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)

model = Sequential()

for i in range(0, num_layers):
    model.add(Dense(500, input_dim=(200,200), activation='relu'))

model.add(Dense(units=120, activation='softmax'))

model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False),
                  metrics=['accuracy', 'categorical_crossentropy', 'mae'])

model.fit(X_train, y_train, epochs=500, validation_data=(X_val, y_val), batch_size=512,
              callbacks=[tensorboard_callback])

classification = model.predict(test_in)

