![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Debian](https://img.shields.io/badge/Debian-D70A53?style=for-the-badge&logo=debian&logoColor=white)

# Tuberculosis

The purpose of this project is to check how well a computer vision model built from scratch performs against pre- trained model (VGG16, VGG19, ResNet, InceptionV3 …). Datasets are X-ray images of patient chests with tuberculosis. Interestingly, the model built from scratch performed very well, achieving an accuracy of 97% on the validation dataset and an accuracy of 95% on the test dataset. The best score was achieved by VGG19 (more than 99% on the validation dataset and test dataset) after using transfer learning techniques – fine tuning.
    
Two methods are mainly used here in order to achieve our goal. Transfer Learning techniques (Feature Extraction && Fine Tuning) and the other is about building a classfier model from scratch (CNN from scratch).

#### The dataset can be downloaded from kaggle: https://www.kaggle.com/tawsifurrahman/tuberculosis-tb-chest-xray-dataset


## Result of the experiment (Alphabetical Order)
### CNN from scratch:
    - Validation accuracy: 97.35 %
    - Validation loss: 8.34 %

    - Test accuracy: 95.63 %
    - Test loss: 12.82 %
    
### DenseNet201:
    - Validation accuracy: 98.12 %
    - Validation loss: 5.20 %
    
    - Test accuracy: 97.50 %
    - Test loss: 7.21 %

### EfficientNetB3:
    - Validation accuracy: 93.82 %
    - Validation loss: 18.21 %

    - Test accuracy: 95.63
    - Test loss: 15.45 %
    
### InceptionV3 - Fine Tuned:
    -Validation accuracy: 97.06 %
    -Validation loss: 7.16 %

    -Test accuracy: 98.75 %
    -Test loss : 4.83 %
    
### ResNet50V2:
    - Validation accuracy: 97.94 %
    - Validation loss: 5.04 %

    - Test accuracy: 98.12 %
    - Test loss : 5.58 %
    
### VGG16 - Fine Tuned:
    - Validation Accuracy: 99.37 %
    - Validation Loss: 0.85 %

    - Test Accuracy: 99.37 %
    - Test Loss: 0.99 %

### VGG19 - Fine Tuned:
    - Validation accuracy: 99.85 %
    - Validation loss: 1.71 %

    - Test accuracy: 99.37 %
    - Test loss: 3.30 %
