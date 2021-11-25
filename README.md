# Tuberculosis
This project is about:
    - Building several tuberculosis disease classifier.
    - Watching how each model performs on the dataset and which one performs the best.
    
Two methods are mainly used here in order to achieve our goal. Transfer Learning techniques (Feature Extraction && Fine Tuning) and the other is about building a classfier model from scratch (CNN from scratch).

#### The dataset can be downloaded from kaggle: https://www.kaggle.com/tawsifurrahman/tuberculosis-tb-chest-xray-dataset


## Result of the experiment
### CNN from scratch:
    - Val accuracy: 97.35 %
    - Val loss: 8.34 %

    - Test accuracy: 95.63 %
    - Test loss: 12.82 %
    
### DenseNet201 (Transfer Learning Feature-Extraction):
    - Val accuracy: 98.12 %
    - Val loss: 5.20 %
    
    - Test accuracy: 97.50 %
    - Test loss: 7.21 %

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
    - Validation Accuracy: 99.37 %
    - Validation Loss: 0.85 %

    - Test Accuracy: 99.37 %
    - Test Loss: 0.99 %

### VGG19 (Transfer Learning Fine-Tuning):
    - Validation accuracy: 99.85 %
    - Validation loss: 1.71 %

    - Test accuracy: 99.37 %
    - Test loss: 3.30 %
    
    
# ACKNOWLEDGEMENT
Thanks to Tawsifur Rahman for this dataset https://www.kaggle.com/tawsifurrahman
