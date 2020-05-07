from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow_core.python.keras import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

new_input = Input(shape=(200,200,3))

model = Sequential()
model.add(ResNet50(weights='imagenet', include_top=False, input_tensor=new_input))
model.trainable = False
model.add(Flatten())
model.add(Dense(120))
print(model.summary())