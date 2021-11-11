from tensorflow.keras.layers import RandomZoom, RandomRotation
from tensorflow.keras import Sequential

data_augmentation = Sequential([
	RandomZoom(0.2),
    RandomRotation(0.1),
])

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy


class Tuber():
    def model(self,input):
        self.x = data_augmentation(input)
        self.x = Rescaling(1./255)(self.x)
        self.x = Conv2D(64,3,activation="relu",padding="same",strides=(2,2))(self.x)
        self.x = MaxPooling2D()(self.x)
        self.x = Conv2D(128,3,activation="relu",padding="same",strides=(2,2))(self.x)
        self.x = Conv2D(128,3,activation="relu",padding="same",strides=(2,2))(self.x)
        self.x = Conv2D(256,3,activation="relu",padding="same",strides=(2,2))(self.x)
        self.x = MaxPooling2D()(self.x)
        self.x = Flatten()(self.x)
        self.x = Dense(128,activation="relu")(self.x)
        self.x = Dropout(0.2,seed=123)(self.x)
        self.x = Dense(64,activation="relu")(self.x)
        self.x = Dropout(0.2,seed=123)(self.x)
        self.outputs = Dense(2,activation="sigmoid")(self.x)
        self.model = Model(input,self.outputs,name="Tuber")
        return self.model

tuber = Tuber()
model = tuber.model(Input(shape=(224,224,3)))
model.summary()
model.compile(RMSprop(),SparseCategoricalCrossentropy(),metrics=["accuracy"])