import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from zipfile import ZipFile

random.seed(1337)
np.random.seed(1337)
tf.random.set_seed(1337)

def unzip(nm):
	with ZipFile("archive.zip","r") as zip:
		zip.extractall()

unzip("archive.zip")


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	"TB_Chest_Radiography_Database",
	validation_split = 0.2,
	subset = "training",
	seed = 1337,
	shuffle = True,
	image_size = (224,224),
	batch_size = 32,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	"TB_Chest_Radiography_Database",
	validation_split = 0.2,
	subset = "validation",
	seed = 1337,
	shuffle = True,
	image_size = (224,224),
	batch_size = 32,
)

val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches//5)
val_ds = val_ds.skip(val_batches//5)

class_names = train_ds.class_names
print(class_names)

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


from tensorflow.keras.layers.experimental.preprocessing import RandomZoom, RandomRotation

data_augmentation = tf.keras.Sequential([
	RandomZoom(0.2),
	RandomRotation(0.1),
])


from tensorflow.keras.applications import resnet_v2, ResNet50V2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import Input, Model

rescale = resnet_v2.preprocess_input

base_model = ResNet50V2(input_shape=(224,224,3),include_top=False,weights="imagenet")
base_model.trainable = False
base_model.summary()


class Transfer_ResNet50V2():
	def model(self,input):
		self.x = data_augmentation(input)
		self.x = rescale(self.x)
		self.x = base_model(self.x,training=False)
		self.x = GlobalAveragePooling2D()(self.x)
		self.x = Dropout(0.2,seed=1337)(self.x)
		self.outputs = Dense(2,activation="sigmoid")(self.x)
		self.model = Model(input,self.outputs,name="Transfer_ResNet50V2")
		return self.model

T = Transfer_ResNet50V2()
model = T.model(Input(shape=(224,224,3)))
model.summary()
model.compile(Adam(),SparseCategoricalCrossentropy(),metrics=["accuracy"])

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

if __name__=="__main__":
	loss0,accuracy0 = model.evaluate(val_ds)
	print("Initial loss: {:.2f} %".format(100*loss0))
	print("Initial accuracy: {:.2f} %".format(100*accuracy0))
	checkpoint = ModelCheckpoint("tuberculosis.hdf5",save_weights_only=False,monitor="val_loss",save_best_only=True)
	model.fit(train_ds,epochs=10,validation_data=val_ds,callbacks=[checkpoint])
	best = load_model("tuberculosis.hdf5")
	val_loss,val_accuracy = best.evaluate(val_ds)
	test_loss,test_accuracy = best.evaluate(test_ds)
	print("\nVal accuracy: {:.2f} %".format(100*val_accuracy))
	print("Val loss: {:.2f} %".format(100*val_loss))
	print("\nTest accuracy: {:.2f} %".format(100*test_accuracy))
	print("Test loss : {:.2f} %".format(100*test_loss))
