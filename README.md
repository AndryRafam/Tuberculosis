# Tuberculosis
This little project is about classifying tuberculosis (X-ray imagery).

#### The dataset can be downloaded from kaggle: https://www.kaggle.com/tawsifurrahman/tuberculosis-tb-chest-xray-dataset

## We are essentially using two methods in order to achieve our goal.
    - The first one is to build a CNN (Convolutional Neural Network) from scratch with data augmentation.
    - The second is to use transfer learning technique (feature extraction) with data augmentation.

## Result of the experiment
### CNN from scratch:
    - Test accuracy: 96.25 %
    - Test loss: 10.61 %
    
### EfficientNetB3 (Transfer Learning):
    - Test accuracy: 98.12
    - Test loss: 6.77 %
    
### ResNet50V2 (Transfer Learning):
    - Test accuracy: 99.37 %
    - Test loss : 5.64 %

### VGG19 (Transfer Learning):
    - Test accuracy: 97.50 %
    - Test loss: 7.55 %
    
# ACKNOWLEDGEMENT
This little project would not have been possible without Kaggle.
