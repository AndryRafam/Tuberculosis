import matplotlib.pyplot as plt
from zipfile import ZipFile

def unzip(nm):
    with ZipFile("archive.zip","r") as zip:
        zip.extractall()

unzip("archive.zip")


from tensorflow.keras.preprocessing import image_dataset_from_directory

train_ds = image_dataset_from_directory(
    directory = "TB_Chest_Radiography_Database",
    validation_split = 0.2,
    subset = "training",
    seed = 1234,
    shuffle = True,
    image_size = (224,224),
    batch_size = 32,
)

val_ds = image_dataset_from_directory(
    directory = "TB_Chest_Radiography_Database",
    validation_split = 0.2,
    subset = "validation",
    seed = 1234,
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


from tensorflow.data import AUTOTUNE

Autotune = AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=Autotune)
val_ds = val_ds.prefetch(buffer_size=Autotune)
test_ds = test_ds.prefetch(buffer_size=Autotune)


import api
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

if __name__=="__main__":
    checkpoint = [
        ModelCheckpoint("tuberculosis.h5",save_weights_only=False,monitor="val_accuracy",save_best_only=True)
    ]
    api.model.fit(train_ds,epochs=7,validation_data=val_ds,callbacks=checkpoint)
    best = load_model("tuberculosis.h5")
    val_loss,val_accuracy = best.evaluate(val_ds)
    test_loss,test_accuracy = best.evaluate(test_ds)
    print("\nVal accuracy: {:.2f} %".format(100*val_accuracy))
    print("Val loss: {:.2f} %".format(100*val_loss))
    print("\nTest accuracy: {:.2f} %".format(100*test_accuracy))
    print("Test loss: {:.2f} %".format(100*test_loss))