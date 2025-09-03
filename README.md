# ChestXray-Classifier (Normal, Pneumonia, Tuberculosis)
A classifier ML model to determine and differentiate chest xrays based on pneumonia , tuberculosis and normal.
This project implements a deep learning model based on ResNet50 to classify chest X-ray images into three categories:
- **Normal**
- **Pneumonia**
- **Tuberculosis**

## Dataset
The dataset was split into **70% training, 15% validation, and 15% testing** across all three classes.  
Preprocessing included:
- Resizing to '224x224'
- Normalization (rescale 1./255)
- Data augmentation (rotation, zoom, horizontal flip)

## Model
- **Architecture**: ResNet50 (pre-trained on ImageNet)
- **Added layers**: GlobalAveragePooling + Dense(256, relu) + Dropout(0.5) + Dense(3, softmax)
- **Loss**: categorical_crossentropy  
- **Optimizer**: Adam  
- **Metrics**: accuracy, precision, recall, f1-score

## Evaluation
- Classification report with precision, recall, f1-score
- Confusion matrix
- Accuracy/Loss curves

## Inference
Run inference on a single X-ray image using the trained model