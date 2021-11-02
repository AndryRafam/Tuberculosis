from tensorflow.keras import Sequential
from tensorflow.keras.layers import RandomZoom, RandomRotation

data_augmentation = Sequential([
    RandomZoom(0.2),
    RandomRotation(0.1),
])


from tensorflow.keras.applications import efficientnet, EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import Input, Model

rescale = efficientnet.preprocess_input

base_model = EfficientNetB3(input_shape=(256,256,3),include_top=False,weights="imagenet")
base_model.trainable = False
base_model.summary()

class Transfer_Efficient():
    def model(self,input):
        self.x = data_augmentation(input)
        self.x = rescale(self.x)
        self.x = base_model(self.x,training=False)
        self.x = GlobalAveragePooling2D()(self.x)
        self.x = Dropout(0.2,seed=1337)(self.x)
        self.outputs = Dense(2,activation="sigmoid")(self.x)
        self.model = Model(input,self.outputs,name="Transfer_EfficientB3")
        return self.model

TEB3 = Transfer_Efficient()
model = TEB3.model(Input(shape=(256,256,3)))
model.summary()
model.compile(Adam(),SparseCategoricalCrossentropy(),metrics=["accuracy"])