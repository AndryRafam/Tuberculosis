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
    - Validation accuracy: 93.82 %
    - Validation loss: 18.21 %

    - Test accuracy: 95.63
    - Test loss: 15.45 %
    
### ResNet50V2 (Transfer Learning Feature-Extraction):
    - Validation accuracy: 97.94 %
    - Validation loss: 5.04 %

    - Test accuracy: 98.12 %
    - Test loss : 5.58 %
    
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
Thanks to Tawsifur Rahman for this dataset https://www.kaggle.com/tawsifurrahman
