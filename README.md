# Tuberculosis

The purpose of this project is to check how well a computer vision model built from scratch performs against pre-trained model such as VGG16, VGG19, ResNet … on the tuberculosis dataset. Interestingly, the model built from scratch performed very well, achieving an accuracy of 97% on the validation dataset and an accuracy of 95% on the test dataset. The best score was achieved by VGG19 (more than 99% on the validation dataset and test dataset) after using transfer learning techniques – fine tuning.
    
Two methods are mainly used here in order to achieve our goal. Transfer Learning techniques (Feature Extraction && Fine Tuning) and the other is about building a classfier model from scratch (CNN from scratch).

#### The dataset can be downloaded from kaggle: https://www.kaggle.com/tawsifurrahman/tuberculosis-tb-chest-xray-dataset


## Result of the experiment (Alphabetical Order)
### CNN from scratch:
    - Validation accuracy: 97.35 %
    - Validation loss: 8.34 %

    - Test accuracy: 95.63 %
    - Test loss: 12.82 %
    
### DenseNet201 (Transfer Learning Feature-Extraction):
    - Validation accuracy: 98.12 %
    - Validation loss: 5.20 %
    
    - Test accuracy: 97.50 %
    - Test loss: 7.21 %

### EfficientNetB3 (Transfer Learning Feature-Extraction):
    - Validation accuracy: 93.82 %
    - Validation loss: 18.21 %

    - Test accuracy: 95.63
    - Test loss: 15.45 %
    
### InceptionV3:
    -Validation accuracy: 97.35 %
    -Validation loss: 7.02 %

    -Test accuracy: 98.75 %
    -Test loss : 4.90 %
    
### ResNet50V2 (Transfer Learning Feature-Extraction):
    - Validation accuracy: 97.94 %
    - Validation loss: 5.04 %

    - Test accuracy: 98.12 %
    - Test loss : 5.58 %
    
### VGG16 (Transfer Learning Fine-Tuning):
    - Validation Accuracy: 99.37 %
    - Validation Loss: 0.85 %

    - Test Accuracy: 99.37 %
    - Test Loss: 0.99 %

### VGG19 (Transfer Learning Fine-Tuning):
    - Validation accuracy: 99.85 %
    - Validation loss: 1.71 %

    - Test accuracy: 99.37 %
    - Test loss: 3.30 %
