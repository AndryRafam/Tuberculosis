import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from zipfile import ZipFile


def unzip(nm):
    with ZipFile(nm,"r") as zip:
        zip.extractall()

unzip("archive.zip")

random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "TB_Chest_Radiography_Database",
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    shuffle = True,
    image_size = (224,224),
    batch_size = 32,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "TB_Chest_Radiography_Database",
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    shuffle = True,
    image_size = (224,224),
    batch_size = 32,
)

from tensorflow.data.experimental import cardinality

val_batches = cardinality(val_ds)
test_ds = val_ds.take(val_batches//5)
val_ds = val_ds.skip(val_batches//5)

class_names = train_ds.class_names

plt.figure(figsize=(12,12))
for images,labels in train_ds.take(1):
    for i in range(4):
        ax = plt.subplot(2,2,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

from tensorflow.keras.layers import RandomZoom, RandomRotation

data_augmentation = tf.keras.Sequential([
	RandomZoom(0.2),
    RandomRotation(0.1),
])

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


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
model.compile(Adam(),SparseCategoricalCrossentropy(),metrics=["accuracy"])

if __name__=="__main__":
    initial_epochs = 50
    loss0,accuracy0 = model.evaluate(val_ds)
    checkpoint = ModelCheckpoint("tuberculosis.hdf5",save_weights_only=False,monitor="val_accuracy",save_best_only=True)
    model.fit(train_ds,epochs=initial_epochs,validation_data=val_ds,callbacks=[checkpoint])
    best = load_model("tuberculosis.hdf5")
    val_loss,val_accuracy = best.evaluate(val_ds)
    test_loss,test_accuracy = best.evaluate(test_ds)
    print("\nVal accuracy: {:.2f} %".format(100*val_accuracy))
    print("Val loss: {:.2f} %".format(100*val_loss))
    print("\nTest accuracy: {:.2f} %".format(100*test_accuracy))
    print("Test loss: {:.2f} %".format(100*test_loss))
