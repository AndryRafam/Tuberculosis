import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from zipfile import ZipFile
from tensorflow.keras.preprocessing import image_dataset_from_directory

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def unzip(nm):
    with ZipFile(nm,"r") as zip:
        zip.extractall()

unzip("archive.zip")

train_ds = image_dataset_from_directory(
    directory = "TB_Chest_Radiography_Database",
    validation_split = 0.2,
    subset = "training",
    seed = 42,
    shuffle = True,
    image_size = (224,224),
    batch_size = 32,
)

val_ds = image_dataset_from_directory(
    directory = "TB_Chest_Radiography_Database",
    validation_split = 0.2,
    subset = "validation",
    seed = 42,
    shuffle = True,
    image_size = (224,224),
    batch_size = 32,
)


from tensorflow.data.experimental import cardinality

val_batches = cardinality(val_ds)
test_ds = val_ds.take(val_batches//5)
val_ds = val_ds.take(val_batches//5)

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


import api

if __name__=="__main__":
    checkpoint = tf.keras.callbacks.ModelCheckpoint("tuberculosis.hdf5",save_weights_only=False,save_best_only=True,monitor="val_accuracy")
    api.model.fit(train_ds,epochs=5,validation_data=val_ds,callbacks=[checkpoint])
    best = tf.keras.models.load_model("tuberculosis.hdf5")
    val_loss,val_acc = best.evaluate(val_ds)
    test_loss,test_acc = best.evaluate(test_ds)
    print("\nVal Accuracy: {:.2f} %".format(100*val_acc))
    print("Val Loss: {:.2f} %".format(100*val_loss))
    print("\nTest Accuracy: {:.2f} %".format(100*test_acc))
    print("Test Loss: {:.2f} %".format(100*test_loss))
