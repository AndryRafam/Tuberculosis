import tensorflow as tf
from tensorflow.keras import Sequential

data_augmentation = Sequential([
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomRotation(0.1),
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
