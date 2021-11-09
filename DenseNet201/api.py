from tensorflow.keras.layers import RandomZoom, RandomRotation
from tensorflow.keras import Sequential

data_augmentation = Sequential([
    RandomZoom(0.2),
    RandomRotation(0.1),
])


from tensorflow.keras.applications import densenet, DenseNet201
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy

rescale = densenet.preprocess_input

base_model = DenseNet201(input_shape=(224,224,3),include_top=False,weights="imagenet")
base_model.trainable = False
base_model.summary()

def model(input):
    x = data_augmentation(input)
    x = rescale(x)
    x = base_model(x,training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2,seed=1234)(x)
    output = Dense(2,activation="softmax")(x)
    model = Model(input,output,name="Transfer_DenseNet201")
    return model

model = model(Input(shape=(224,224,3)))
model.summary()
model.compile(RMSprop(),SparseCategoricalCrossentropy(),metrics=["accuracy"])
