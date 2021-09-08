import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from zipfile import ZipFile

random.seed(1337)
np.random.seed(1337)
tf.random.set_seed(1337)

with ZipFile("archive.zip","r") as zip:
	zip.extractall()

BATCH_SIZE = 32
IMG_SIZE = (224,224)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	"TB_Chest_Radiography_Database",
	validation_split = 0.2,
	subset = "training",
	seed = 1337,
	shuffle = True,
	image_size = IMG_SIZE,
	batch_size = BATCH_SIZE,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	"TB_Chest_Radiography_Database",
	validation_split = 0.2,
	subset = "validation",
	seed = 1337,
	shuffle = True,
	image_size = IMG_SIZE,
	batch_size = BATCH_SIZE,
)

val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches//5)
val_ds = val_ds.skip(val_batches//5)

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10,10))
for images,_ in train_ds.take(1):
	for i in range(9):
		ax = plt.subplot(3,3,i+1)
		plt.imshow(images[i].numpy().astype("uint8"))
		plt.axis("off")
plt.show()

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
	tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
])

plt.figure(figsize=(10,10))
for images,_ in train_ds.take(1):
	for i in range(9):
		ax = plt.subplot(3,3,i+1)
		augmented_images = data_augmentation(images)
		plt.imshow(augmented_images[0].numpy().astype("uint8"))
		plt.axis("off")
plt.show()

from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import Input, Model

preprocess_input = resnet_v2.preprocess_input

IMG_SHAPE = IMG_SIZE+(3,)
base_model = ResNet50V2(input_shape=IMG_SHAPE,include_top=False,weights="imagenet")
image_batch,label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)

base_model.trainable = False
base_model.summary()
feature_batch_average = GlobalAveragePooling2D()(feature_batch)

prediction_layer = Dense(2,activation="sigmoid")
prediction_batch = prediction_layer(feature_batch_average)

class Transfer_ResNet():
	def model(self,y):
		self.x = data_augmentation(y)
		self.x = preprocess_input(self.x)
		self.x = base_model(self.x,training=False)
		self.x = GlobalAveragePooling2D()(self.x)
		self.x = Dropout(0.2,seed=1337)(self.x)
		self.outputs = prediction_layer(self.x)
		self.model = Model(y,self.outputs)
		return self.model

m = Transfer_ResNet()
model = m.model(tf.keras.Input(shape=IMG_SHAPE))
model.summary()
model.compile(Adam(),SparseCategoricalCrossentropy(),metrics=["accuracy"])

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

if __name__=="__main__":
	loss0,accuracy0 = model.evaluate(val_ds)
	print("Initial loss: {:.2f} %".format(100*loss0))
	print("Initial accuracy: {:.2f} %".format(100*accuracy0))
	checkpoint = ModelCheckpoint("tuberculosis.hdf5",save_weights_only=False,monitor="val_loss",save_best_only=True)
	model.fit(train_ds,epochs=7,validation_data=val_ds,callbacks=[checkpoint])
	best = load_model("tuberculosis.hdf5")
	loss,accuracy = best.evaluate(test_ds)
	print("\nTest accuracy: {:.2f} %".format(100*accuracy))
	print("Test loss : {:.2f} %".format(100*loss))
