from tensorflow.keras import Sequential
from tensorflow.keras.layers import RandomZoom, RandomRotation

data_augmentation = Sequential([
    RandomZoom(0.2),
    RandomRotation(0.1),
])

from tensorflow.keras.applications import inception_v3, InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import Input, Model

rescale = inception_v3.preprocess_input

base_model = InceptionV3(input_shape=(299,299,3),include_top=False,weights="imagenet")

for layer in base_model.layers:
    layer.trainable = False

def model(input):
    x = data_augmentation(input)
    x = rescale(x)
    x = base_model(x,training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(2)(x)
    model = Model(input,outputs,name="Transfer_InceptionV3")
    return model

model = model(Input(shape=(299,299,3)))
model.summary()
model.compile(RMSprop(),SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])

