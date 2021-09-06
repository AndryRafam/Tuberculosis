import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from zipfile import ZipFile
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom

with ZipFile("archive.zip","r") as zip:
    zip.extractall()

random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

batch_size = 32
img_size = (224,224)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "TB_Chest_Radiography_Database",
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    shuffle = True,
    image_size = img_size,
    batch_size  = batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "TB_Chest_Radiography_Database",
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    shuffle = True,
    image_size = img_size,
    batch_size = batch_size,
)

val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches//5)
val_ds = val_ds.skip(val_batches//5)

plt.figure(figsize=(10,10))
for images,_ in train_ds.take(1):
    for i in range(12):
        ax = plt.subplot(3,4,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")
plt.show()

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
	RandomZoom(0.2),
])

plt.figure(figsize=(10,10))
for images,_ in train_ds.take(1):
    for i in range(12):
        ax = plt.subplot(3,4,i+1)
        augmented_images = data_augmentation(images)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
plt.show()

from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

input_shape = img_size+(3,)

class Tuber():
    def model(self,y):
        self.x = data_augmentation(y)
        self.x = Rescaling(1./255)(self.x)
        self.x = Conv2D(64,3,activation="relu",padding="same",strides=(2,2))(self.x)
        self.x = MaxPooling2D()(self.x)
        self.x = Conv2D(128,3,activation="relu",padding="same",strides=(2,2))(self.x)
        self.x = Conv2D(128,3,activation="relu",padding="same",strides=(2,2))(self.x)
        self.x = Conv2D(256,3,activation="relu",padding="same",strides=(2,2))(self.x)
        self.x = MaxPooling2D()(self.x)
        self.x = Flatten()(self.x)
        self.x = Dense(64,activation="relu")(self.x)
        self.x = Dropout(0.4,seed=123)(self.x)
        self.x = Dense(32,activation="relu")(self.x)
        self.x = Dropout(0.4,seed=123)(self.x)
        self.outputs = Dense(2,activation="sigmoid")(self.x)
        self.model = tf.keras.Model(y,self.outputs)
        return self.model

tuber = Tuber()
model = tuber.model(tf.keras.Input(shape=input_shape))
model.summary()
model.compile(tf.keras.optimizers.Adam(),tf.keras.losses.SparseCategoricalCrossentropy(),metrics=["accuracy"])

if __name__=="__main__":
    initial_epochs = 50
    loss0,accuracy0 = model.evaluate(val_ds)
    checkpoint = tf.keras.callbacks.ModelCheckpoint("tuberculosis.hdf5",save_weights_only=False,monitor="val_accuracy",save_best_only=True)
    model.fit(train_ds,epochs=initial_epochs,validation_data=val_ds,callbacks=[checkpoint])
    best = tf.keras.models.load_model("tuberculosis.hdf5")
    loss,accuracy = best.evaluate(test_ds)
    print("\nTest accuracy: {:.2f} %".format(100*accuracy))
    print("Test loss: {:.2f} %".format(100*loss))