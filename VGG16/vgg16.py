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


from tensorflow.keras import Sequential

data_augmentation = Sequential([
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
])


rescale = tf.keras.applications.vgg16.preprocess_input

base_model = tf.keras.applications.VGG16(input_shape=(224,224,3),include_top=False,weights="imagenet")
base_model.trainable = True
base_model.summary()


for layer in base_model.layers:
    if layer.name == "block3_pool":
        break
    layer.trainable = False

class Transfer_VGG16():
    def model(self,input):
        self.x = data_augmentation(input)
        self.x = rescale(self.x)
        self.x = base_model(self.x,training=False)
        self.x = tf.keras.layers.GlobalAveragePooling2D()(self.x)
        self.x = tf.keras.layers.Flatten()(self.x)
        self.x = tf.keras.layers.Dense(128,activation="relu")(self.x)
        self.x = tf.keras.layers.Dropout(0.2,seed=42)(self.x)
        self.x = tf.keras.layers.Dense(64,activation="relu")(self.x)
        self.x = tf.keras.layers.Dropout(0.2,seed=42)(self.x)
        self.output = tf.keras.layers.Dense(2,activation="sigmoid")(self.x)
        self.model = tf.keras.Model(input,self.output,name="Transfer_VGG16")
        return self.model

TFVGG16 = Transfer_VGG16()
model = TFVGG16.model(tf.keras.Input(shape=(224,224,3)))
model.summary()
model.compile(tf.keras.optimizers.Adam(1e-5),tf.keras.losses.SparseCategoricalCrossentropy(),metrics=["accuracy"])


if __name__=="__main__":
    checkpoint = tf.keras.callbacks.ModelCheckpoint("tuberculosis.hdf5",save_weights_only=False,save_best_only=True,monitor="val_accuracy")
    model.fit(train_ds,epochs=5,validation_data=val_ds,callbacks=[checkpoint])
    best = tf.keras.models.load_model("tuberculosis.hdf5")
    val_loss,val_acc = best.evaluate(val_ds)
    test_loss,test_acc = best.evaluate(test_ds)
    print("\nVal Accuracy: {:.2f} %".format(100*val_acc))
    print("Val Loss: {:.2f} %".format(100*val_loss))
    print("\nTest Accuracy: {:.2f} %".format(100*test_acc))
    print("Test Loss: {:.2f} %".format(100*test_loss))