from tensorflow.keras import Sequential
from tensorflow.keras.layers import RandomZoom, RandomRotation

data_augmentation = Sequential([
    RandomZoom(0.2),
    RandomRotation(0.1),
])

from tensorflow.keras.applications import vgg19, VGG19
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import Input, Model

rescale = vgg19.preprocess_input

base_model = VGG19(input_shape=(224,224,3),include_top=False,weights="imagenet")
base_model.trainable = True
base_model.summary()

fine_tune_at = 11

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

class Transfer_VGG19():
    def model(self,input):
        self.x = data_augmentation(input)
        self.x = rescale(self.x)
        self.x = base_model(self.x,training=False)
        self.x = Flatten()(self.x)
        self.x = Dense(128,activation="relu")(self.x)
        self.x = Dropout(0.4,seed=42)(self.x)
        self.x = Dense(64,activation="relu")(self.x)
        self.x = Dropout(0.4,seed=42)(self.x)
        self.outputs = Dense(2,activation="sigmoid")(self.x)
        self.model = Model(input,self.outputs,name="Transfer_VGG19")
        return self.model

TVGG19 = Transfer_VGG19()
model = TVGG19.model(Input(shape=(224,224,3)))
model.summary()
model.compile(Adam(learning_rate=1e-5),SparseCategoricalCrossentropy(),metrics=["accuracy"])