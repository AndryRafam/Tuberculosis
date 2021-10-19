# Tuberculosis
This little project is about classifying tuberculosis (X-ray imagery).

#### The dataset can be downloaded from kaggle: https://www.kaggle.com/tawsifurrahman/tuberculosis-tb-chest-xray-dataset

In order to achieve our goal, we are going to use two methods:

    - 1) Building a CNN (Convolutional Neural Network) from scratch with data augmentation.
    - 2) Transfer learning techniques with data augmentation.

## Result of the experiment
### CNN from scratch:
    - Validation accuracy: 97.50 %
    - Validation loss: 8.76 %
    
    - Test accuracy: 96.88 %
    - Test loss: 12.27 %
    
### EfficientNetB3 (Transfer Learning Feature-Extraction):
    - Validation accuracy: 96.62 %
    - Validation loss: 10.21 %
    
    - Test accuracy: 98.75
    - Test loss: 9.56 %
    
### ResNet50V2 (Transfer Learning Feature-Extraction):
    - Validation accuracy: 98.09 %
    - Validation loss: 4.79 %

    - Test accuracy: 97.50 %
    - Test loss : 4.94 %

### VGG16 (Transfer Learning Fine-Tuning):
    - Validation Accuracy: 100.00 %
    - Validation Loss: 0.84 %

    - Test Accuracy: 100.00 %
    - Test Loss: 1.03 %

### VGG19 (Transfer Learning Fine-Tuning):
    - Validation accuracy: 99.56 %
    - Validation loss: 1.67 %
    
    - Test accuracy: 99.37 %
    - Test loss: 3.12 %
    
    
# ACKNOWLEDGEMENT
Thanks to Tawsifur Rahman for this dataset. https://www.kaggle.com/tawsifurrahman
