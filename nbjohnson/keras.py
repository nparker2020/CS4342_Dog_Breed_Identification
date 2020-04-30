from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import datetime

num_layers = 3

data_train = []
labels_train = []
data_validate = []
labels_validate = []

# graph changes over epochs
logdir = "logs/scalars/" + datetime.time().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)

model = Sequential()

for i in range(0, num_layers)
    model.add(Dense(500, input_dim=(200,200), activation='relu'))

model.add(Dense(units=120, activation='softmax'))

model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False),
                  metrics=['accuracy', 'categorical_crossentropy', 'mae'])

model.fit(data_train, labels_train, epochs=500, validation_data=(data_validate, labels_validate), batch_size=512,
              callbacks=[tensorboard_callback])

# classification = model.predict(data_test)

